"""
Golden evaluation dataset: question/expected_answer/expected_sources triples.
Mirrors the golden-eval approach used in the jd_resume_match project.
Covers all 4 sample docs: leave policy, travel policy, benefits (PDF),
and IT security policy (DOCX).
"""

GOLDEN_DATASET = [
    # --- Leave policy ---
    {
        "question": "How many days of paid annual leave do full-time employees get?",
        "expected_answer": "18 days per calendar year.",
        "expected_sources": [("company_leave_policy.txt", 1)],
    },
    {
        "question": "How much notice must an employee give before resigning?",
        "expected_answer": "A notice period of 60 days.",
        "expected_sources": [("company_leave_policy.txt", 1)],
    },
    # --- Travel policy ---
    {
        "question": "What class of flight is approved for domestic travel under 6 hours?",
        "expected_answer": "Economy class.",
        "expected_sources": [("company_travel_policy.txt", 1)],
    },
    {
        "question": "Who needs to approve international travel?",
        "expected_answer": "VP-level approval is required regardless of duration.",
        "expected_sources": [("company_travel_policy.txt", 1)],
    },
    # --- Benefits PDF ---
    {
        "question": "How many years of continuous service are needed to become eligible for gratuity?",
        "expected_answer": "Five continuous years of service.",
        "expected_sources": [("company_benefits_compensation.pdf", 2)],
    },
    {
        "question": "What is the referral bonus amount for a successful hire?",
        "expected_answer": "Rs. 25,000, paid after the referred candidate completes 90 days of employment.",
        "expected_sources": [("company_benefits_compensation.pdf", 3)],
    },
    {
        "question": "Within how many days must an employee submit an expense reimbursement claim?",
        "expected_answer": "Within 45 days of the expense being incurred.",
        "expected_sources": [("company_benefits_compensation.pdf", 4)],
    },
    # --- IT Security DOCX ---
    {
        "question": "How quickly must a lost or stolen company device be reported?",
        "expected_answer": "Within 2 hours of discovery.",
        "expected_sources": [("company_it_security_policy.docx", 1)],
    },
    {
        "question": "How many days per week can an employee work remotely without prior approval?",
        "expected_answer": "Up to 3 days per week, if the role is designated remote-eligible.",
        "expected_sources": [("company_it_security_policy.docx", 1)],
    },
    # --- Cross-document synthesis ---
    {
        "question": "If an employee wants to take personal leave right before an official business trip, how is that handled?",
        "expected_answer": (
            "It must be requested separately through the standard annual leave "
            "process; it is not covered under the travel policy."
        ),
        "expected_sources": [("company_travel_policy.txt", 1), ("company_leave_policy.txt", 1)],
    },
    {
        "question": "Compare how much advance notice is needed for taking leave versus booking official travel.",
        "expected_answer": (
            "Leave requests need at least 3 working days' notice, while official "
            "travel bookings need at least 5 working days' notice."
        ),
        "expected_sources": [("company_leave_policy.txt", 1), ("company_travel_policy.txt", 1)],
    },
    {
        "question": (
            "Compare the reporting/notice timelines across policies: expense claims, "
            "lost devices, and security incidents."
        ),
        "expected_answer": (
            "Expense reimbursement claims must be filed within 45 days; lost or "
            "stolen devices must be reported within 2 hours; active security "
            "incidents must be reported within 1 hour."
        ),
        "expected_sources": [
            ("company_benefits_compensation.pdf", 4),
            ("company_it_security_policy.docx", 1),
        ],
    },
]


def get_golden_dataset():
    return GOLDEN_DATASET