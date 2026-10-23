"""Every prompt the agent uses, in one place.

Collected here rather than scattered through the node definitions because
prompts are the part of this system most often edited, most often edited by
someone who is not reading the graph wiring, and hardest to review inside a
closure.

Two of them are calibrated and should not be edited casually:

**The router prompt is a two-sided calibration.** It must send external factual
questions to `kb` -- otherwise the model answers from memory with no evidence --
while keeping greetings *and questions about the assistant itself* on `direct`.
An earlier version pushed everything informational to `kb`; `"who are you?"`
then retrieved nothing, fell through to web search, and answered from an
unrelated web page. Re-test both classes after any edit.

**The grader prompt decides what "good" means.** It is what stops a chunk from
the right general subject being treated as an answer. The similarity gate cannot
do this job: a wrong-document chunk has scored as high as 0.521.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

ROUTER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You route messages for an internal HR assistant.\n"
            "Answer 'direct' for messages that need no evidence lookup: "
            "greetings, thanks, farewells, small talk, and questions about who "
            "or what this assistant is and what it can help with.\n"
            "Answer 'kb' for every message seeking a fact -- company policy, "
            "benefits, leave, pay, expenses, conduct, and equally any question "
            "about the outside world. Downstream stages check whether the "
            "knowledge base covers it and fall back to web search when it does "
            "not, so routing to 'kb' is never wasted.\n"
            "Never route an external factual question to 'direct': that makes "
            "the assistant answer from memory with no evidence behind it, which "
            "is the one thing it must not do.",
        ),
        ("human", "{question}"),
    ]
)

GRADER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You judge whether the supplied evidence can answer the question.\n"
            "Answer 'good' only if the evidence states the specific facts the "
            "question asks for.\n"
            "Answer 'weak' if the evidence is off-topic, only generally "
            "related, or omits the specific detail requested.\n"
            "Judge the evidence as given. Do not use outside knowledge, and do "
            "not credit evidence for being merely on the same broad subject.",
        ),
        ("human", "Question:\n{question}\n\nEvidence:\n{evidence}"),
    ]
)

KB_ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You answer employee questions from internal HR policy.\n"
            "Use ONLY the supplied context. Never invent a number, date or "
            "entitlement that is not in it.\n"
            "If the context covers only part of the question, answer that part "
            "and say plainly which part it does not cover.\n"
            "Write for a colleague, not a lawyer: lead with the answer, then the "
            "detail. Cite the source document names you used.\n"
            "End with: Source: Private KB.",
        ),
        ("human", "Question:\n{question}\n\nPrivate KB context:\n{context}"),
    ]
)

WEB_ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Internal HR policy did not cover this question, so the answer comes "
            "from a public web search.\n"
            "Use ONLY the supplied search results. Never invent details.\n"
            "Open by saying this is general public information and not this "
            "company's policy, so the reader does not mistake it for an "
            "entitlement. Include the most useful URLs.\n"
            "End with: Source: Web Search.",
        ),
        ("human", "Question:\n{question}\n\nWeb search context:\n{context}"),
    ]
)

DIRECT_ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an internal HR assistant. This message needs no policy "
            "lookup. Reply in one or two warm, natural sentences. Do not invent "
            "policy. If the person seems to want something, invite the question.",
        ),
        ("human", "{question}"),
    ]
)

REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite an employee's question so it retrieves better from an HR "
            "policy knowledge base and from web search.\n"
            "Preserve the original intent exactly. Prefer the formal vocabulary "
            "a policy document would use over casual phrasing. Keep it one "
            "sentence.\n"
            "Do not answer the question. Return only the rewritten query.",
        ),
        ("human", "{question}"),
    ]
)

CONTEXTUALIZE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite a follow-up question from an HR chat so it stands on its "
            "own, using the conversation above only to resolve what the "
            "question refers to.\n"
            "Replace pronouns and elliptical phrases -- 'it', 'that', 'what "
            "about contractors?' -- with the thing they point at.\n"
            "**If the question already stands on its own, return it unchanged.** "
            "Do not add detail the employee did not ask for, do not merge in the "
            "previous question, and do not answer anything. Return only the "
            "question.",
        ),
        ("human", "Conversation so far:\n{history}\n\nFollow-up: {question}"),
    ]
)

# --- Few-shot grading --------------------------------------------------------
# The same instruction as GRADER_PROMPT, plus four labelled examples.
#
# **Separate template rather than an edit to the calibrated one.** GRADER_PROMPT
# is what the 0.15 gate and the 21/21 labelled set were measured against;
# changing it in place would invalidate both numbers and there would be no
# way to tell which change moved which figure. This is the A/B: switch it on
# with `FEWSHOT_GRADING=1`, re-run `scripts/eval_decisions.py`, and compare.
#
# The four examples are the four corners of the decision, which is the whole
# reason there are four rather than ten. One of each is enough to show the
# boundary; more mostly adds tokens to every graded question, and the grader
# runs once or twice per answer.
#
#   1. right document, right fact                  -> good
#   2. right subject, wrong specific               -> weak   (the hard one)
#   3. adjacent subject, plausible-looking         -> weak
#   4. several chunks, one of which has the fact   -> good
GRADER_EXAMPLES: list[tuple[str, str, str]] = [
    (
        "How many days of paid time off do full-time employees get?",
        "[1] source=leave-and-time-off.md\nFull-time employees accrue 20 days of "
        "paid time off per calendar year, accruing monthly from the start date.",
        "good",
    ),
    (
        "How much notice do I have to give to resign?",
        "[1] source=onboarding-and-offboarding.md\nOffboarding begins when HR "
        "receives written notice. The manager completes an exit checklist and IT "
        "revokes access on the final working day.",
        "weak",
    ),
    (
        "Can I claim a taxi home after working late?",
        "[1] source=remote-and-hybrid-work.md\nEmployees working from home may "
        "claim a one-off home office allowance within their first month.",
        "weak",
    ),
    (
        "When are expenses paid out?",
        "[1] source=attendance-and-working-hours.md\nCore hours are 10:00 to "
        "16:00 local time.\n\n[2] source=expense-reimbursement.md\nApproved "
        "expenses are paid with the next payroll run after approval.",
        "good",
    ),
]

GRADER_PROMPT_FEWSHOT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You judge whether the supplied evidence can answer the question.\n"
            "Answer 'good' only if the evidence states the specific facts the "
            "question asks for.\n"
            "Answer 'weak' if the evidence is off-topic, only generally "
            "related, or omits the specific detail requested.\n"
            "Judge the evidence as given. Do not use outside knowledge, and do "
            "not credit evidence for being merely on the same broad subject.\n\n"
            "Worked examples:\n"
            + "\n\n".join(
                f"Question: {question}\nEvidence:\n{evidence}\nGrade: {grade}"
                for question, evidence, grade in GRADER_EXAMPLES
            ),
        ),
        ("human", "Question:\n{question}\n\nEvidence:\n{evidence}"),
    ]
)


# --- Re-ranking --------------------------------------------------------------

RERANK_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You score how well each numbered passage answers one question.\n"
            "10 means the passage directly states the fact asked for. 5 means it "
            "is on the right subject but does not state the fact. 0 means it is "
            "unrelated.\n"
            "Return exactly one score per passage, in the order given. Do not "
            "merge, reorder, skip or explain them -- the caller matches your "
            "numbers to its passages by position, so a missing score silently "
            "shifts every passage after it.",
        ),
        ("human", "Question:\n{question}\n\nPassages:\n{passages}"),
    ]
)


# --- Model routing -----------------------------------------------------------

COMPLEXITY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You judge how much reasoning one HR question needs, so a cheaper "
            "model can take the easy ones.\n"
            "simple: a greeting, a thank-you, or a single fact with one obvious "
            "home ('what are core hours?').\n"
            "standard: an ordinary policy question needing one lookup and a "
            "careful read ('how is PTO paid out when I leave?').\n"
            "complex: spans two or more policy areas, needs arithmetic, asks "
            "for a comparison, or turns on an edge case ('I am a contractor "
            "going on parental leave mid-notice-period -- what am I owed?').\n"
            "Judge the question, not how interesting it is. Most questions are "
            "standard.",
        ),
        ("human", "{question}"),
    ]
)


# --- Splitting ---------------------------------------------------------------

SPLIT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You decide whether one employee question is really several.\n"
            "Split it ONLY when it asks about two or more separate policy areas "
            "that different documents would cover -- leave and payroll, "
            "benefits and conduct. One retrieval over a question like that "
            "fills its top-k with whichever area the corpus covers better, and "
            "the other half goes unanswered.\n"
            "Do NOT split a single question that merely has several clauses, "
            "asks for a number and its deadline, or names two things inside one "
            "policy area.\n"
            "When you split, write each part as a question that stands on its "
            "own. Two or three parts, never more.",
        ),
        ("human", "{question}"),
    ]
)


# --- The ReAct loop ----------------------------------------------------------
# One prompt, rendered per turn with the tool catalogue and the transcript so
# far. The tool list is injected rather than hardcoded because the registry is
# what decides which tools a *caller* may use -- an employee and an HR admin see
# different catalogues, and a prompt naming a tool the registry would refuse is
# a model being invited to fail.

REACT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an internal HR assistant that answers from evidence, one "
            "step at a time.\n\n"
            "Each turn: say in one sentence what you now know and what you need "
            "next, then either call ONE tool or give the final answer. Never "
            "both.\n\n"
            "Rules that matter more than being helpful:\n"
            "- Never state a number, date or entitlement that a tool did not "
            "return. If the evidence does not cover it, say which part it does "
            "not cover.\n"
            "- Search the private knowledge base before the public web. "
            "Company policy outranks a general article, and an article that "
            "contradicts policy is wrong here.\n"
            "- Do not call the same tool twice with the same arguments. If a "
            "search came back empty, change the wording or try another tool.\n"
            "- Cite the source document names you used.\n"
            "- When a tool reports that it needs human approval, stop and say "
            "so plainly. Do not work around it.\n\n"
            "Available tools:\n{tools}",
        ),
        (
            "human",
            "Question:\n{question}\n\n"
            "Work so far (empty on the first step):\n{transcript}\n\n"
            "Budget left: {budget}. Give your next step.",
        ),
    ]
)

REACT_FORCED_ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You ran out of budget part way through answering an HR question. "
            "Write the best answer the evidence below supports, and say plainly "
            "in one sentence what you were unable to check. Do not invent the "
            "missing part, and do not apologise at length.\n"
            "Cite the source document names you used.",
        ),
        ("human", "Question:\n{question}\n\nEvidence gathered:\n{transcript}"),
    ]
)


# --- History compaction ------------------------------------------------------
# Used by `compact()` in `app/agent/memory.py` when a conversation outgrows its
# character budget. It only ever sees the turns *outside* the recent window,
# because a summarised turn cannot resolve a pronoun with precision and that is
# the one job history has here.

HISTORY_SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Summarise the earlier part of an HR chat in at most three "
            "sentences, for an assistant that still has the most recent turns "
            "verbatim.\n"
            "Keep what the employee asked about and any fact they stated about "
            "themselves -- their location, employment type, working pattern. "
            "Drop pleasantries and drop the assistant's own wording.\n"
            "Do not answer anything and do not add detail nobody said.",
        ),
        ("human", "{transcript}"),
    ]
)


# --- Supervisor and workers --------------------------------------------------

SUPERVISOR_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are the supervisor of a small HR support team. You do not "
            "answer questions yourself; you decide which specialist looks at "
            "one next, and when the team has enough to answer.\n\n"
            "Your specialists:\n"
            "- policy: leave, time off, attendance, working hours, remote and "
            "hybrid work\n"
            "- benefits: insurance, medical cover, allowances, wellbeing\n"
            "- payroll: pay, salary, payslips, tax, expenses, reimbursement\n"
            "- workplace: conduct, grievance, data privacy, employee records, "
            "forms and requests\n"
            "- web: public, non-company information, and only after the private "
            "specialists have come back empty\n"
            "- general: anything that does not obviously belong to one of the "
            "above\n"
            "- escalate: a person must decide. Use it when the question needs a "
            "judgement the documents do not contain, concerns someone's "
            "individual case, alleges misconduct, or asks for an exception.\n\n"
            "Set done=true as soon as the findings below answer the question. A "
            "further handoff that adds nothing costs the employee time and the "
            "company a call.\n"
            "Do not send the same question to the same specialist twice. If one "
            "came back with nothing, either try a different specialist or "
            "finish.",
        ),
        (
            "human",
            "Question:\n{question}\n\n"
            "Specialists already consulted: {consulted}\n"
            "Findings so far:\n{findings}\n\n"
            "Who looks at this next, or are you done?",
        ),
    ]
)

WORKER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are the {worker} specialist on an internal HR support team. "
            "You have been handed one question and the evidence retrieved from "
            "your own documents.\n\n"
            "Report what your documents say, in two or three sentences. Use "
            "ONLY the evidence given. If it does not cover the question, say so "
            "in one sentence and stop -- a specialist reporting 'my documents "
            "do not cover this' is doing their job, and guessing to look useful "
            "is the one failure the supervisor cannot detect.\n"
            "Name the source documents you used. Do not write a closing or a "
            "greeting; the supervisor composes the reply.",
        ),
        ("human", "Question:\n{question}\n\nYour evidence:\n{context}"),
    ]
)

COMPOSE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are the supervisor of an HR support team, writing the single "
            "reply the employee will read. Your specialists have reported "
            "below.\n\n"
            "Merge their findings into one answer. Lead with the answer, then "
            "the detail. Keep each specialist's facts and drop their "
            "scaffolding.\n"
            "Where two specialists disagree, say so explicitly and name both "
            "documents rather than silently picking one -- a confidently merged "
            "contradiction is the worst answer this team can produce.\n"
            "Where nobody covered part of the question, say which part.\n"
            "Never add a fact no specialist reported. Cite the source document "
            "names. End with: Source: Private KB.",
        ),
        (
            "human",
            "Question:\n{question}\n\nSpecialist findings:\n{findings}",
        ),
    ]
)

ESCALATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "An HR question is being escalated to a person. Write the note that "
            "person will read first.\n"
            "Two short paragraphs: what the employee asked, and what the "
            "assistant established or failed to establish from policy. Then one "
            "line naming the decision the person has to make.\n"
            "Do not answer the question, do not speculate about the employee's "
            "situation, and do not include anything the evidence does not say.",
        ),
        (
            "human",
            "Question:\n{question}\n\n"
            "Why it was escalated: {reason}\n\n"
            "What was found:\n{findings}",
        ),
    ]
)


# --- Extractive fallback -----------------------------------------------------
# Not a prompt: the answer composed with no language model at all. See
# `app/services/extractive.py`.


# --- Canned answers ----------------------------------------------------------
# Not prompts, but the same category of user-visible copy.

INSUFFICIENT_ANSWER = (
    "I could not find enough reliable evidence in the private knowledge base or "
    "in web search results to answer this confidently. Please rephrase the "
    "question, or point me at the specific policy document that covers it."
)

GENERATION_FAILED = (
    "I could not produce an answer just now because the language model was "
    "unreachable. The question was understood; please try again shortly."
)

# Shown when the input guard refuses a question. Deliberately says what to do
# next and does not quote the rule that fired: a refusal that explains exactly
# which pattern matched is a refusal that teaches someone how to get around it.
BLOCKED_BY_GUARD = (
    "I can't help with that one. I answer questions about this company's HR "
    "policy -- leave, pay, benefits, conduct, working arrangements. Ask me one "
    "of those and I'll look it up."
)

# Shown when a question needs a person. Names the next step rather than leaving
# the employee at a dead end, because the alternative to this message is them
# asking a manager the same question tomorrow.
ESCALATED_ANSWER = (
    "This one needs a person, not a policy lookup. I've logged it for the HR "
    "team with what I found, and someone will pick it up. Nothing you've told "
    "me has gone anywhere else."
)

# Appended when a run stopped on its budget. One sentence, and it says the
# answer is partial: an answer that ran out of room and does not say so is
# indistinguishable from a complete one that happened to be short.
BUDGET_STOPPED_NOTE = (
    "\n\n_This answer is partial: I reached my limit for this question before "
    "checking everything. Ask again for the part I missed and I'll start fresh._"
)

# Prefix for an answer assembled from retrieved text with no language model --
# the no-key path. It leads rather than trails because the thing a reader needs
# to know about this answer is that nothing wrote it.
EXTRACTIVE_PREFIX = (
    "**Straight from the policy documents** (no language model is configured on "
    "this deployment, so this is the relevant text rather than a written "
    "answer):\n\n"
)
