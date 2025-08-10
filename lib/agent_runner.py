from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from .tracing import SQLiteTracer
from .db import DB_PATH
from lib.tools.agent_tools import *
import os
import sqlite3
from datetime import datetime
import threading

def run_agent(session_id: str, message_id: str, user_input: str, conversation_context: str = ""):
    try:
        print(f"Starting agent execution for message {message_id[:8]}... in session {session_id[:8]}...")
        print(f"User input: {user_input[:100]}...")
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE messages 
            SET status = 'running' 
            WHERE message_id = ?
        ''', (message_id,))
        conn.commit()
        conn.close()
        tracer = SQLiteTracer(message_id)
        from pydantic import SecretStr
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.1,
            callbacks=[tracer],
            # openai_api_key=os.environ.get("OPENAI_API_KEY"),
            api_key=SecretStr(os.environ["OPENAI_API_KEY"])
        )
        tools = [
            web_search,
            get_current_time,
            calculate_math,
            analyze_data,
            read_file,
            write_file,
            list_directory,
            execute_python_code,
            get_weather,
            translate_text,
            url_fetch,
            json_processor
        ]
        for t in tools:
            t.callbacks = [tracer]
        prompt_template = hub.pull("hwchase17/react")
        agent = create_react_agent(llm, tools, prompt_template)
        executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            callbacks=[tracer],
            handle_parsing_errors=True,
            max_iterations=5,
            max_execution_time=30,
            return_intermediate_steps=False
        )
        current_date = datetime.now().strftime("%B %d, %Y")
        current_day = datetime.now().strftime("%A")
        main_context = conversation_context + user_input
        system_context = f"""
\n[System Information - Current date: {current_day}, {current_date}]\n\n[Operating Guidelines: TEXT-ONLY environment]\n- Provide text-based responses only\n- No visual plots, charts, or graphs\n- Use text tables/lists for data visualization\n\n[CRITICAL TOOL USAGE RULES - READ CAREFULLY]\n\nIMPORTANT: You MUST follow the ReAct format exactly:\n\n1. TO USE A TOOL:\nThought: I need to calculate something\nAction: tool_name\nAction Input: raw_input_value\n\n2. TO GIVE FINAL ANSWER (END THE CONVERSATION):\nFinal Answer: Your complete response here.\n\nEXAMPLES:\n✅ CORRECT TOOL USAGE:\nThought: I need to calculate the square root of 16\nAction: calculate_math\nAction Input: sqrt(16)\n\n✅ CORRECT MATH INPUT:\nAction Input: 1 * ln(0.9) / ln(0.9)\n\n❌ WRONG MATH INPUT (causes errors):\nAction Input: \"1 * ln(0.9) / ln(0.9)\"  # With quotes and comments\nAction Input: \"1 * ln(0.9) / ln(0.9)\"  # First interval calculation\n\n✅ CORRECT FINAL ANSWER:\nFinal Answer: The square root of 16 is 4.\n\n❌ WRONG - CAUSES INFINITE LOOPS:\nThought: Here's my answer... (WITHOUT Action: or Final Answer:)\n\nRULES:\n- NEVER include markdown backticks (```) in Action Input!\n- After getting your answer, use \"Final Answer:\" to end\n- NEVER have \"Thought:\" without either \"Action:\" or \"Final Answer:\" following it\n- If a tool fails 3+ times, try a different approach!\n"""
        full_input = main_context + system_context
        try:
            result = executor.invoke({"input": full_input})
            final_output = result.get("output", "")
            is_agent_error = (
                not final_output or 
                len(final_output.strip()) < 10 or
                "iteration limit" in final_output.lower() or
                "time limit" in final_output.lower() or
                "stopped due to" in final_output.lower()
            )
            is_format_loop = (
                final_output and
                ("Invalid Format:" in final_output or
                 "Missing 'Action:' after 'Thought:'" in final_output or
                 final_output.count("Invalid Format") > 1)
            )
            is_irrelevant = (
                final_output and 
                len(final_output.strip()) < 100 and 
                ("current date is" in final_output.lower() or
                 final_output.strip().startswith("The current date") or
                 final_output.strip().startswith("Today is"))
            )
            if is_agent_error or is_irrelevant or is_format_loop:
                if not final_output or len(final_output.strip()) < 10:
                    final_output = "I encountered technical difficulties while processing your request. Please try rephrasing your question or ask something else."
                elif is_irrelevant:
                    final_output = "I apologize, but I gave an irrelevant response. Please let me try to answer your question properly."
                    raise Exception(f"Agent gave irrelevant response: {final_output}")
                elif is_format_loop:
                    final_output = "I encountered a formatting error loop and couldn't complete the task properly. Please try rephrasing your question."
                    raise Exception(f"Agent stuck in format loop: {final_output}")
                else:
                    raise Exception(f"Agent execution failed: {final_output}")
            print(f"Agent execution completed successfully for message {message_id[:8]}...")
            print(f"Output length: {len(final_output)} characters")
        except Exception as agent_error:
            print(f"Agent execution error for message {message_id[:8]}: {str(agent_error)}")
            try:
                fallback_prompt = f"""Please respond to this query in a helpful way. Use only text - no visual elements.\n\n{conversation_context}Current query: {user_input}\n\nProvide a relevant, helpful response that addresses the user's question in the context of our conversation."""
                from app import client
                fallback_response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{
                        "role": "user",
                        "content": fallback_prompt
                    }],
                    max_tokens=500
                )
                if fallback_response.choices and fallback_response.choices[0].message:
                    final_output = fallback_response.choices[0].message.content or "I'm having technical difficulties. Please try again."
                else:
                    final_output = "I'm experiencing technical difficulties. Please try again later."
                print(f"Used context-aware fallback response for message {message_id[:8]}")
            except Exception as fallback_error:
                print(f"Fallback also failed for message {message_id[:8]}: {str(fallback_error)}")
                final_output = "I'm experiencing technical difficulties and cannot process your request at the moment. Please try again later."
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE messages 
                SET status = 'error', content = ?, completed_at = CURRENT_TIMESTAMP
                WHERE message_id = ?
            ''', (final_output, message_id))
            cursor.execute('''
                UPDATE sessions 
                SET updated_at = CURRENT_TIMESTAMP
                WHERE session_id = ?
            ''', (session_id,))
            conn.commit()
            conn.close()
            return
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE messages 
            SET status = 'completed', content = ?, completed_at = CURRENT_TIMESTAMP
            WHERE message_id = ?
        ''', (final_output, message_id))
        cursor.execute('''
            UPDATE sessions 
            SET updated_at = CURRENT_TIMESTAMP
            WHERE session_id = ?
        ''', (session_id,))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"UNEXPECTED ERROR in run_agent for message {message_id}: {str(e)}")
        print(f"Session: {session_id}, User input: {user_input[:100]}...")
        user_friendly_error = "I encountered an unexpected error while processing your request. Please try again later."
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE messages 
            SET status = 'error', content = ?, completed_at = CURRENT_TIMESTAMP
            WHERE message_id = ?
        ''', (user_friendly_error, message_id))
        cursor.execute('''
            UPDATE sessions 
            SET updated_at = CURRENT_TIMESTAMP
            WHERE session_id = ?
        ''', (session_id,))
        conn.commit()
        conn.close()
