"""Application-owned write authorization, independent of the model's plan."""

import re
from uuid import UUID

from personal_assistant.schemas import Action

DESTRUCTIVE_COMMANDS = {
    "delete_note": r"delete\s+note|remove\s+note|удали\s+заметку|удалить\s+заметку",
    "forget": r"forget(?:\s+fact)?|забудь(?:\s+факт)?|удали\s+факт",
    "cancel_event": r"cancel\s+event|delete\s+event|отмени\s+событие|отменить\s+событие",
}

TARGETED_WRITES = {"update_note", "complete_task", "update_event"}
ARTICLES = (
    r"(?:(?:a|an|the|my|new|another|one|two|three|four|five|[1-5]"
    r"|мою|мои|новую|эту|две|два|три|одну)\s+)*"
)
OBJECTS = {
    "note": r"(?:notes?|заметк[а-яё]*|записк[а-яё]*)\b",
    "task": r"(?:tasks?|to-dos?|reminders?|задач[а-яё]*|напоминани[а-яё]*)\b",
    "event": r"(?:(?:calendar\s+)?events?|meetings?|appointments?|событи[а-яё]*|встреч[а-яё]*)\b",
}
VERBS = {
    "save_note": (
        r"save|write down|record|take|make|create|add|сохрани(?:ть)?"
        r"|запиши|записать|создай|создать|добавь|добавить"
    ),
    "update_note": (
        r"update|edit|change|append|add|обнови(?:ть)?|измени(?:ть)?"
        r"|добавь|добавить|дополни(?:ть)?"
    ),
    "create_task": (
        r"create|add|make|schedule|создай|создать|добавь|добавить|запланируй|запланировать"
    ),
    "complete_task": r"complete|finish|заверши(?:ть)?|выполни(?:ть)?",
    "create_event": (
        r"schedule|book|create|add|plan|запланируй|запланировать"
        r"|создай|создать|добавь|добавить|назначь|назначить"
    ),
    "update_event": (
        r"move|reschedule|update|edit|change|перенеси|перенести"
        r"|обнови(?:ть)?|измени(?:ть)?|сдвинь|сдвинуть"
    ),
}


def request_clauses(text: str) -> list[str]:
    # Only the imperative header grants authority. Quoted text and saved content
    # after a colon/newline cannot introduce additional commands.
    header = re.split(r"[:\n\r]", text.strip(), maxsplit=1)[0]
    header = re.sub(
        r"^(?:(?:please[,\s]+|can you\s+|could you\s+|пожалуйста[,\s]+|можешь\s+))+",
        "",
        header,
        flags=re.I,
    )
    if re.search(r"\b(?:never|not|don't|don’t|cannot|нельзя|не)\b", header, re.I):
        return []
    parts, start = [], 0
    for match in re.finditer(
        r""""(?:\\.|[^"\\])*(?:"|$)|'(?:\\.|[^'\\])*(?:'|$)|«[^»]*(?:»|$)"""
        r"|(?P<split>\s+(?:and|и)\s+|;\s*)",
        header,
        re.I,
    ):
        if match.group("split"):
            parts.append(header[start : match.start()].strip())
            start = match.end()
    parts.append(header[start:].strip())
    # Share only an explicit creation verb in lists such as "Create a note and a task".
    creation = re.fullmatch(
        r"(create|add|make|создай|создать|добавь|добавить)\s+"
        + ARTICLES
        + "(?:"
        + "|".join(OBJECTS.values())
        + "),?",
        parts[0],
        re.I,
    )
    if creation:
        for index in range(1, len(parts)):
            if re.match(ARTICLES + "(?:" + "|".join(OBJECTS.values()) + ")", parts[index], re.I):
                parts[index] = creation[1] + " " + parts[index]
    return parts


def identifies_target(subject: str, action: Action, target: dict | None) -> bool:
    kind = (
        "note"
        if action.name == "update_note"
        else ("task" if action.name == "complete_task" else "event")
    )
    selectors = []
    try:
        selectors.append(("(?i:" + re.escape(str(UUID(action.resource_id))) + ")", True))
    except ValueError:
        pass
    # Exact, case-sensitive names avoid guessing between differently named records.
    title = target.get("title", "") if target and target.get("unique_title") else ""
    if title:
        literal = re.escape(title)
        selectors.append((rf'''(?:"{literal}"|'{literal}'|«{literal}»|{literal})''', False))
    for selector, is_id in selectors:
        patterns = [
            rf"(?i:{ARTICLES}{OBJECTS[kind]}\s+){selector}",
            rf"(?i:{ARTICLES}){selector}(?i:\s+{OBJECTS[kind]})",
        ]
        if is_id:
            patterns.append(selector)
        # The selector must be the object, not an ID/name mentioned in the update body.
        suffix = r"(?:(?i:\s+(?:to|with|using|as|на|с|как|используя)\b)|[.!]?\s*$)"
        if re.match("(?:" + "|".join(patterns) + ")" + suffix, subject):
            return True
    return False


def authorize_mutation(action: Action, text: str, target: dict | None = None):
    intent = text.strip().casefold()
    if action.name in DESTRUCTIVE_COMMANDS:
        try:
            identifier = str(UUID(action.resource_id))
        except ValueError:
            identifier = ""
        command = (
            rf"(?:please\s+|пожалуйста[,\s]+)?(?:/{action.name}|"
            rf"{DESTRUCTIVE_COMMANDS[action.name]})\s+(?:(?:with\s+)?id\s*[:=]?\s*|с\s+id\s+)?"
            rf"{re.escape(identifier)}[.!]?"
        )
        if not identifier or not re.fullmatch(command, intent):
            raise ValueError(
                f"Write not applied. To authorize this exact operation, send /{action.name} "
                f"{action.resource_id or '<ID>'} as a separate message."
            )
        return
    allowed = False
    for clause in request_clauses(text):
        if action.name == "remember":
            allowed = bool(re.match(r"(?:remember|memorize|запомни(?:ть)?|помни)\b", clause, re.I))
        elif action.name == "create_task" and re.match(
            r"(?:remind\s+me|напомни(?:ть)?)\b", clause, re.I
        ):
            allowed = True
        elif (
            action.name == "save_note"
            and re.fullmatch(r"note|заметка", clause, re.I)
            and re.match(r"(?:note|заметка)\s*:", text.strip(), re.I)
        ):
            allowed = True
        else:
            verb = re.match(r"(?:" + VERBS[action.name] + r")\b\s+", clause, re.I)
            if not verb:
                continue
            subject = clause[verb.end() :]
            kind = (
                "note"
                if action.name.endswith("note")
                else ("task" if action.name.endswith("task") else "event")
            )
            noun = ARTICLES + OBJECTS[kind]
            allowed = bool(re.match(noun, subject, re.I))
            if action.name in TARGETED_WRITES:
                allowed = identifies_target(subject, action, target)
        if allowed:
            break
    if not allowed:
        raise ValueError(
            "Write not applied: explicit user intent could not be established. "
            "Name the note/task/event before ':', and identify edits by exact ID "
            "or exact unique stored title."
        )
    if action.name in {"create_event", "update_event"} and re.search(
        r"\bevery\b|\bweekly\b|\bdaily\b|\bmonthly\b|\byearly\b|\brecurr|кажд|ежеднев|еженед|ежемес",
        intent,
    ):
        raise ValueError("Recurring events are not supported; request a single occurrence")
