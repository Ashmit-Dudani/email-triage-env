"""
data.py - Email dataset with ground-truth labels
Each email has a fixed correct answer so grading is fully deterministic.
"""

from models import Email

# ─────────────────────────────────────────────
# GROUND-TRUTH LABELS
# Keys match email_id values below.
# ─────────────────────────────────────────────
GROUND_TRUTH: dict[str, dict] = {
    "email_001": {"category": "work",     "priority": "high",   "action_type": "reply"},
    "email_002": {"category": "spam",     "priority": "low",    "action_type": "ignore"},
    "email_003": {"category": "work",     "priority": "high",   "action_type": "escalate"},
    "email_004": {"category": "promo",    "priority": "low",    "action_type": "ignore"},
    "email_005": {"category": "personal", "priority": "medium", "action_type": "reply"},
    "email_006": {"category": "work",     "priority": "high",   "action_type": "reply"},
    "email_007": {"category": "spam",     "priority": "low",    "action_type": "ignore"},
    "email_008": {"category": "work",     "priority": "medium", "action_type": "reply"},
    "email_009": {"category": "personal", "priority": "low",    "action_type": "ignore"},
    "email_010": {"category": "work",     "priority": "high",   "action_type": "escalate"},
}

# ─────────────────────────────────────────────
# EMAIL OBJECTS
# ─────────────────────────────────────────────
EMAIL_POOL: list[Email] = [

    Email(
        email_id="email_001",
        subject="Q3 Report — Action Required by EOD",
        body=(
            "Hi, I need the Q3 financial report on my desk before end of day. "
            "The board meeting is tomorrow morning. This is urgent."
        ),
        sender_type="boss",
    ),

    Email(
        email_id="email_002",
        subject="You've WON $1,000,000!!!",
        body=(
            "Congratulations! You have been selected as our lucky winner. "
            "Click here to claim your prize immediately. Limited time offer!"
        ),
        sender_type="unknown",
    ),

    Email(
        email_id="email_003",
        subject="Server outage — production is DOWN",
        body=(
            "URGENT: The main production server has been unreachable for 30 minutes. "
            "Customers cannot access the service. We need a decision on failover NOW."
        ),
        sender_type="client",
    ),

    Email(
        email_id="email_004",
        subject="50% OFF — Flash Sale ends tonight!",
        body=(
            "Don't miss our biggest sale of the year! "
            "Use code FLASH50 at checkout. Shop now before stocks run out."
        ),
        sender_type="promo",
    ),

    Email(
        email_id="email_005",
        subject="Family reunion next weekend",
        body=(
            "Hey! Just confirming you're coming to the reunion on Saturday. "
            "Let me know if you need directions or want to carpool."
        ),
        sender_type="unknown",
    ),

    Email(
        email_id="email_006",
        subject="Contract renewal — please confirm terms",
        body=(
            "Dear team, please review the attached contract renewal for our annual SLA. "
            "We need your sign-off by Friday to avoid a lapse in coverage."
        ),
        sender_type="client",
    ),

    Email(
        email_id="email_007",
        subject="Nigerian prince needs your help",
        body=(
            "Dear friend, I am a prince who needs to transfer $10 million out of my country. "
            "I will give you 30% if you share your bank details."
        ),
        sender_type="unknown",
    ),

    Email(
        email_id="email_008",
        subject="Weekly team standup notes",
        body=(
            "Hi all, attached are the notes from today's standup. "
            "Key items: sprint review on Thursday, design handoff due Wednesday."
        ),
        sender_type="boss",
    ),

    Email(
        email_id="email_009",
        subject="Happy Birthday!",
        body=(
            "Wishing you a wonderful birthday! Hope you have a fantastic day "
            "and a great year ahead. 🎂"
        ),
        sender_type="unknown",
    ),

    Email(
        email_id="email_010",
        subject="Legal notice — compliance deadline tomorrow",
        body=(
            "This is a formal notice that your compliance documentation must be submitted "
            "by 9 AM tomorrow or you will face regulatory penalties. Immediate escalation required."
        ),
        sender_type="client",
    ),
]

# Quick lookup by email_id
EMAIL_BY_ID: dict[str, Email] = {e.email_id: e for e in EMAIL_POOL}
