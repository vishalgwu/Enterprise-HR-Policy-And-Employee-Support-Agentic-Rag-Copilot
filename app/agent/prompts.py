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
