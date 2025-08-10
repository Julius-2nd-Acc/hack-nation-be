import re

def replace_current_request(text, new_request):
    # This regex matches everything between 'Current request:\n' and '\n[System Information'
    return re.sub(
        r'(Current request:\n)(.*?)(\n\[System Information)',
        rf'\1{new_request}\3',
        text,
        flags=re.DOTALL
    )

def replace_before_system_info(text, new_prefix):
    # This regex matches everything from the start up to (but not including) '\n[System Information'
    return re.sub(r'^.*?(?=\n\[System Information)', new_prefix, text, flags=re.DOTALL)