# HR Copilot: How the Agentic RAG Assistant Works

## Purpose
The HR Copilot answers employee questions about internal HR policy. It is an agentic retrieval
augmented generation (RAG) system: rather than always answering from a single retrieval pass, it
grades what it retrieves and decides what to do next.

## Retrieval Step
An employee question is embedded and matched against the private knowledge base, which holds the
internal policy documents in this repository. The retriever returns the top-k most similar chunks
together with their similarity scores.

## Document Grading
Retrieved chunks are graded for relevance before any answer is written. The grader labels each chunk
relevant or not relevant to the question actually asked, rather than trusting the vector similarity
score alone. Similarity is a measure of closeness in embedding space, and a chunk can be the closest
available match while still being useless for the question.

## What Happens If Retrieved Documents Are Not Relevant
If retrieved documents are not relevant, the copilot does not answer from them and does not
hallucinate an answer. Instead the graph takes a fallback branch. First it rewrites the question,
because a poor retrieval is often caused by vocabulary mismatch between how the employee phrased the
question and how the policy is written, and retries retrieval against the private knowledge base
once. If the rewritten query still returns nothing relevant, the copilot escalates to a Tavily web
search for public information, and clearly marks the answer as sourced from the public web rather
than from internal policy. If neither path yields relevant material, the copilot states that it does
not know and routes the employee to a human HR business partner. This grade-then-branch loop is what
makes the system agentic rather than a single-shot RAG pipeline.

## Guardrails
The copilot never invents policy numbers, dates, or entitlements. Every answer drawn from internal
policy cites the source document. Questions touching an individual compensation record, medical
information, immigration status, or an open investigation are always routed to a human.
