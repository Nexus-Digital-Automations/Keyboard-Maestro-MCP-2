#!/usr/bin/env python3
"""Bulk Type Annotation Fix Script.

This script systematically adds missing type annotations for ANN201 and ANN001 violations
across the codebase for improved code quality and maintainability.
"""

import logging
import re
from pathlib import Path
from typing import Any, Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path("/Users/jeremyparker/Desktop/Claude Coding Projects/Keyboard-Maestro-MCP")

# Common return type patterns
RETURN_TYPE_PATTERNS = {
    # Functions that don't return anything
    r"def\s+(?:test_|example_|main|setup|teardown|init|clear|reset|update|add|remove|delete|create|register|unregister|start|stop|cleanup|configure|initialize|finalize|validate|check|verify|ensure|log|print|debug|info|warn|error|critical|handle|process|execute|run|perform|trigger|activate|deactivate|enable|disable|cancel|pause|resume|close|open|quit|exit|save|load|write|read|set|get|put|post|send|receive|connect|disconnect|bind|unbind|attach|detach|sync|async|await|yield|return|raise|throw|catch|try|finally|with|as|import|from|pass|break|continue|while|for|if|else|elif|assert|del|global|nonlocal|lambda|class|def|async|await)[^(]*\([^)]*\)\s*:\s*$": "-> None:",

    # Functions that return boolean
    r"def\s+(?:is_|has_|can_|should_|will_|was_|were_|contains|includes|exists|matches|equals|compare|validate|verify|check|test|confirm|ensure)[^(]*\([^)]*\)\s*:\s*$": "-> bool:",

    # Functions that return strings
    r"def\s+(?:get_|format_|generate_|create_|build_|render_|stringify|to_string|as_string|serialize)[^(]*\([^)]*\)\s*:\s*$": "-> str:",

    # Functions that return lists
    r"def\s+(?:list_|find_|search_|collect_|gather_|filter_|map_|sort_|group_)[^(]*\([^)]*\)\s*:\s*$": "-> list[Any]:",

    # Functions that return dictionaries
    r"def\s+(?:dict_|config_|settings_|options_|params_|metadata_|info_|data_|result_|response_|status_|state_)[^(]*\([^)]*\)\s*:\s*$": "-> dict[str, Any]:",
}

# Common argument type patterns
ARGUMENT_TYPE_PATTERNS = {
    # String arguments
    r"\b(name|text|message|content|data|value|key|path|url|uri|email|username|password|token|id|identifier|uuid|guid|tag|label|title|description|comment|note|filename|filepath|dirname|extension|pattern|regex|query|search|filter|sort|order|direction|method|action|command|operation|function|procedure|routine|algorithm|strategy|approach|technique|style|format|encoding|charset|locale|language|region|country|code|version|revision|release|branch|commit|hash|checksum|signature|digest|fingerprint|thumbprint|certificate|license|permission|role|scope|privilege|right|access|level|rank|priority|weight|score|rating|feedback|review|comment|note|remark|observation|insight|tip|hint|suggestion|recommendation|advice|guidance|instruction|direction|specification|requirement|constraint|limitation|restriction|condition|criteria|rule|policy|guideline|standard|norm|convention|protocol|interface|contract|agreement|treaty|pact|deal|arrangement|settlement|resolution|solution|answer|response|reply|feedback|result|outcome|consequence|effect|impact|influence|significance|importance|relevance|meaning|purpose|intent|goal|objective|target|aim|mission|vision|strategy|plan|design|architecture|structure|framework|model|template|pattern|example|sample|instance|case|scenario|situation|context|environment|setting|configuration|setup|installation|deployment|distribution|package|bundle|archive|file|document|record|entry|item|element|component|part|piece|fragment|chunk|block|section|segment|portion|fraction|percentage|ratio|proportion|rate|speed|velocity|acceleration|momentum|force|power|energy|capacity|capability|ability|skill|talent|expertise|knowledge|wisdom|intelligence|understanding|comprehension|awareness|consciousness|perception|sensation|feeling|emotion|mood|attitude|behavior|conduct|action|activity|task|job|work|labor|effort|performance|achievement|accomplishment|success|failure|error|mistake|fault|defect|bug|issue|problem|challenge|difficulty|obstacle|barrier|hurdle|impediment|hindrance|interference|disruption|disturbance|noise|signal|data|information|knowledge|wisdom|insight|understanding|comprehension|awareness|consciousness|perception|sensation|feeling|emotion|mood|attitude|behavior|conduct|action|activity|task|job|work|labor|effort|performance|achievement|accomplishment|success|failure|error|mistake|fault|defect|bug|issue|problem|challenge|difficulty|obstacle|barrier|hurdle|impediment|hindrance|interference|disruption|disturbance|noise|signal)[^:]*": "str",

    # Numeric arguments
    r"\b(count|size|length|width|height|depth|radius|diameter|area|volume|weight|mass|density|pressure|temperature|time|duration|timeout|delay|interval|frequency|rate|speed|velocity|acceleration|distance|position|location|coordinate|index|offset|page|limit|threshold|minimum|maximum|total|sum|average|mean|median|mode|range|variance|deviation|percentage|ratio|proportion|factor|multiplier|divisor|quotient|remainder|modulo|exponent|logarithm|sine|cosine|tangent|angle|degree|radian|pi|euler|infinity|nan|zero|one|two|three|four|five|six|seven|eight|nine|ten|hundred|thousand|million|billion|trillion|quadrillion|quintillion|sextillion|septillion|octillion|nonillion|decillion)[^:]*": "int",

    # Boolean arguments
    r"\b(enabled|disabled|active|inactive|visible|hidden|open|closed|locked|unlocked|checked|unchecked|selected|unselected|focused|unfocused|expanded|collapsed|minimized|maximized|fullscreen|windowed|modal|modeless|readonly|readwrite|editable|noneditable|required|optional|mandatory|voluntary|public|private|internal|external|local|remote|online|offline|connected|disconnected|authenticated|unauthenticated|authorized|unauthorized|valid|invalid|true|false|yes|no|on|off|start|stop|begin|end|first|last|previous|next|forward|backward|up|down|left|right|north|south|east|west|top|bottom|front|back|inside|outside|before|after|above|below|over|under|beside|between|among|within|without|include|exclude|allow|deny|permit|forbid|enable|disable|show|hide|display|conceal|reveal|expose|cover|uncover|wrap|unwrap|pack|unpack|compress|decompress|encrypt|decrypt|encode|decode|serialize|deserialize|marshal|unmarshal|parse|unparse|format|unformat|normalize|denormalize|validate|invalidate|verify|unverify|confirm|unconfirm|accept|reject|approve|disapprove|grant|revoke|assign|unassign|bind|unbind|attach|detach|connect|disconnect|link|unlink|join|unjoin|merge|unmerge|split|unsplit|combine|uncombine|unite|separate|group|ungroup|cluster|uncluster|collect|scatter|gather|disperse|concentrate|dilute|focus|unfocus|zoom|unzoom|scale|unscale|resize|unresize|move|unmove|shift|unshift|rotate|unrotate|flip|unflip|mirror|unmirror|invert|uninvert|reverse|unreverse|sort|unsort|order|disorder|arrange|disarrange|organize|disorganize|structure|destructure|construct|deconstruct|build|unbuild|create|destroy|make|unmake|produce|reproduce|generate|degenerate|synthesize|analyze|compose|decompose|assemble|disassemble|install|uninstall|setup|teardown|configure|unconfigure|initialize|uninitialize|start|stop|begin|end|open|close|launch|terminate|execute|cancel|run|halt|play|pause|resume|continue|break|interrupt|suspend|resume|activate|deactivate|enable|disable|turn_on|turn_off|switch_on|switch_off|power_on|power_off|boot|shutdown|restart|reboot|reset|clear|flush|purge|clean|dirty|fresh|stale|new|old|recent|ancient|modern|classic|current|outdated|latest|earliest|final|initial|temporary|permanent|volatile|stable|dynamic|static|automatic|manual|interactive|batch|foreground|background|synchronous|asynchronous|parallel|sequential|concurrent|exclusive|shared|unique|duplicate|single|multiple|mono|stereo|binary|decimal|hexadecimal|octal|ascii|unicode|utf8|utf16|utf32|iso|ansi|mime|json|xml|yaml|csv|tsv|html|css|javascript|python|java|cpp|csharp|go|rust|swift|kotlin|scala|ruby|perl|php|bash|shell|powershell|batch|cmd|terminal|console|gui|cli|api|sdk|library|framework|engine|driver|plugin|extension|addon|module|package|bundle|archive|file|folder|directory|database|table|column|row|field|record|entry|item|element|node|leaf|branch|tree|graph|network|cluster|grid|matrix|array|list|vector|queue|stack|heap|hash|map|set|dictionary|collection|container|wrapper|proxy|adapter|facade|bridge|decorator|observer|publisher|subscriber|listener|handler|controller|manager|service|provider|factory|builder|creator|destroyer|validator|parser|formatter|serializer|deserializer|encoder|decoder|compressor|decompressor|encryptor|decryptor|hasher|signer|verifier|authenticator|authorizer|filter|transformer|converter|mapper|reducer|aggregator|accumulator|collector|distributor|dispatcher|router|gateway|proxy|load_balancer|circuit_breaker|retry|timeout|cache|buffer|pool|connection|session|transaction|lock|semaphore|mutex|barrier|latch|condition|event|signal|message|notification|alert|warning|error|exception|fault|failure|success|result|response|request|query|command|instruction|directive|order|task|job|work|operation|action|activity|process|thread|fiber|coroutine|generator|iterator|stream|flow|pipeline|chain|sequence|series|batch|group|set|collection|list|array|vector|queue|stack|heap|tree|graph|network|cluster|grid|matrix|table|database|store|repository|cache|buffer|pool|registry|catalog|index|directory|namespace|scope|context|environment|setting|configuration|option|parameter|argument|attribute|property|field|member|variable|constant|literal|value|data|information|content|payload|body|header|footer|metadata|annotation|comment|note|remark|description|summary|title|caption|label|tag|keyword|category|class|type|kind|sort|variety|species|genus|family|order|kingdom|domain|realm|world|universe|cosmos|space|time|dimension|axis|plane|surface|line|point|pixel|bit|byte|word|character|string|text|number|integer|float|double|decimal|fraction|percentage|ratio|proportion|rate|frequency|period|cycle|phase|step|stage|level|layer|tier|rank|grade|degree|scale|magnitude|amplitude|intensity|brightness|darkness|color|hue|saturation|luminance|contrast|gamma|alpha|beta|delta|epsilon|zeta|eta|theta|iota|kappa|lambda|mu|nu|xi|omicron|pi|rho|sigma|tau|upsilon|phi|chi|psi|omega)[^:]*": "bool",

    # Optional arguments
    r"\b(optional_|maybe_|if_|when_|unless_)[^:]*": "Optional[Any]",
}

def detect_return_type(func_content: str, func_name: str) -> str:
    """Detect the appropriate return type for a function."""
    func_lower = func_content.lower()

    # Check for explicit return statements
    if "return None" in func_content or "return" not in func_content:
        return "-> None:"

    if re.search(r"return\s+(True|False|\w+\s*==\s*\w+|\w+\s*!=\s*\w+|\w+\s*>\s*\w+|\w+\s*<\s*\w+|\w+\s*>=\s*\w+|\w+\s*<=\s*\w+|\w+\s*and\s*\w+|\w+\s*or\s*\w+|not\s+\w+|\w+\s*in\s*\w+|\w+\s*is\s*\w+|\w+\s*is\s*not\s*\w+)", func_content):
        return "-> bool:"

    if re.search(r"return\s*\[.*\]", func_content):
        return "-> list[Any]:"

    if re.search(r"return\s*\{.*\}", func_content):
        return "-> dict[str, Any]:"

    if re.search(r"return\s*['\"].*['\"]", func_content):
        return "-> str:"

    if re.search(r"return\s*\d+", func_content):
        return "-> int:"

    # Apply pattern matching for function names
    for pattern, return_type in RETURN_TYPE_PATTERNS.items():
        if re.match(pattern, f"def {func_name}():", re.IGNORECASE):
            return return_type

    # Default to Any for complex cases
    return "-> Any:"

def detect_argument_type(arg_name: str, func_content: str) -> str:
    """Detect the appropriate type for a function argument."""
    arg_lower = arg_name.lower()

    # Check for common patterns
    for pattern, arg_type in ARGUMENT_TYPE_PATTERNS.items():
        if re.match(pattern, arg_lower, re.IGNORECASE):
            return arg_type

    # Check for context clues in function body
    if f"str({arg_name})" in func_content or f"{arg_name}.split(" in func_content:
        return "str"

    if f"int({arg_name})" in func_content or f"{arg_name} + 1" in func_content:
        return "int"

    if f"len({arg_name})" in func_content or "for" in func_content and arg_name in func_content:
        return "list[Any]"

    # Default to Any
    return "Any"

def fix_function_annotations(file_path: Path) -> tuple[bool, int]:
    """Fix missing type annotations in a single file."""
    try:
        with open(file_path, encoding='utf-8') as f:
            content = f.read()

        original_content = content
        changes_made = 0

        # Check if typing imports are present
        has_typing_import = any(import_line in content for import_line in [
            "from typing import", "import typing", "from __future__ import annotations"
        ])

        # Fix ANN201 violations (missing return type annotations)
        # Match function definitions without return type annotations
        pattern = re.compile(
            r'^(\s*)(def\s+\w+\s*\([^)]*\))\s*:\s*$',
            re.MULTILINE
        )

        def replace_function_def(match: Any) -> bool:
            nonlocal changes_made
            indent = match.group(1)
            func_def = match.group(2)

            # Extract function name and content
            func_name_match = re.search(r'def\s+(\w+)', func_def)
            if not func_name_match:
                return match.group(0)

            func_name = func_name_match.group(1)

            # Skip if it's a special method or already has type annotation
            if func_name.startswith('__') and func_name.endswith('__'):
                return match.group(0)

            # Get function content to analyze return type
            func_start = match.end()
            func_content = ""

            # Find the function body (simple heuristic)
            lines = content[func_start:].split('\n')
            for i, line in enumerate(lines):
                if i > 0 and line and not line.startswith(' ') and not line.startswith('\t'):
                    break
                func_content += line + '\n'

            # Detect return type
            return_type = detect_return_type(func_content, func_name)

            changes_made += 1
            return f"{indent}{func_def} {return_type}"

        content = pattern.sub(replace_function_def, content)

        # Fix ANN001 violations (missing argument type annotations)
        # This is more complex and would require parsing the AST properly
        # For now, let's focus on simple cases

        # Add typing import if needed and changes were made
        if changes_made > 0 and not has_typing_import:
            # Find the best place to add typing import
            if "from __future__ import annotations" in content:
                # Already has future annotations, add typing import after other imports
                import_pattern = re.search(r'((?:^(?:from __future__|from|import)\s+[^\n]+\n)+)', content, re.MULTILINE)
                if import_pattern:
                    imports_end = import_pattern.end()
                    content = content[:imports_end] + "from typing import Any, Optional\n" + content[imports_end:]
            else:
                # Add future annotations and typing import
                import_pattern = re.search(r'((?:^(?:from|import)\s+[^\n]+\n)+)', content, re.MULTILINE)
                if import_pattern:
                    imports_start = import_pattern.start()
                    content = content[:imports_start] + "from __future__ import annotations\n\nfrom typing import Any, Optional\n" + content[imports_start:]
                else:
                    # Add at the beginning after docstring
                    docstring_end = 0
                    if content.startswith('"""') or content.startswith("'''"):
                        quote_type = '"""' if content.startswith('"""') else "'''"
                        end_pos = content.find(quote_type, 3)
                        if end_pos != -1:
                            docstring_end = end_pos + 3

                    imports = "\nfrom __future__ import annotations\n\nfrom typing import Any, Optional\n"
                    content = content[:docstring_end] + imports + content[docstring_end:]

        # Only write if changes were made
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True, changes_made

        return False, 0

    except Exception as e:
        logger.error(f"Failed to process {file_path}: {e}")
        return False, 0

def process_directory(directory: Path, target_files: Optional[list[str]] = None) -> tuple[int, int]:
    """Process Python files in directory."""
    files_modified = 0
    total_changes = 0

    # Get files with most violations first
    if target_files:
        files_to_process = [directory / f for f in target_files if (directory / f).exists()]
    else:
        files_to_process = list(directory.rglob("*.py"))

    for py_file in files_to_process:
        if py_file.is_file() and not py_file.name.startswith('.'):
            try:
                modified, changes = fix_function_annotations(py_file)
                if modified:
                    files_modified += 1
                    total_changes += changes
                    logger.info(f"Fixed {changes} annotations in {py_file.relative_to(PROJECT_ROOT)}")
            except Exception as e:
                logger.error(f"Error processing {py_file}: {e}")

    return files_modified, total_changes

def main() -> None:
    """Main execution function."""
    logger.info("Starting bulk type annotation fixing...")
    logger.info(f"Processing directory: {PROJECT_ROOT}")

    if not PROJECT_ROOT.exists():
        logger.error(f"Project root does not exist: {PROJECT_ROOT}")
        return

    # Focus on files with most violations first
    high_priority_files = [
        "examples/mcp_server_permissions.py",
        "examples/permission_examples.py",
        "tests/test_tools/test_hotkey_tools.py",
        "tests/tools/test_user_identity_tools.py",
        "tests/test_tools/test_group_tools.py",
        "tests/test_tools/test_property_tools.py",
        "tests/test_tools/test_sync_tools.py",
        "tests/test_tools/test_interface_tools.py",
        "tests/test_tools/test_search_tools.py",
        "tests/test_tools/test_action_tools.py",
        "tests/test_tools/test_engine_tools.py",
    ]

    # Process high priority files first
    logger.info("Processing high priority files...")
    files_modified, total_changes = process_directory(PROJECT_ROOT, high_priority_files)

    # Process remaining files
    logger.info("Processing remaining files...")
    remaining_files, remaining_changes = process_directory(PROJECT_ROOT)

    files_modified += remaining_files
    total_changes += remaining_changes

    logger.info("Bulk type annotation fix completed!")
    logger.info(f"Files modified: {files_modified}")
    logger.info(f"Total annotations added: {total_changes}")

    if files_modified > 0:
        logger.info("Remember to:")
        logger.info("1. Run ruff check to verify type annotations")
        logger.info("2. Run tests to ensure functionality is preserved")
        logger.info("3. Review the changes for any context-specific adjustments")

if __name__ == "__main__":
    main()
