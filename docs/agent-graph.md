# Agent graph

Generated from the compiled graph by `python scripts/render_graph.py`. Do not edit by hand.

Solid arrows are unconditional edges; dotted arrows are branches taken on a structured decision.

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	contextualize(contextualize)
	route_question(route_question)
	retrieve_kb(retrieve_kb)
	grade_kb_evidence(grade_kb_evidence)
	search_web(search_web)
	grade_web_evidence(grade_web_evidence)
	rewrite_query(rewrite_query)
	generate_from_kb(generate_from_kb)
	generate_from_web(generate_from_web)
	direct_answer(direct_answer)
	answer_insufficient(answer_insufficient)
	__end__([<p>__end__</p>]):::last
	__start__ --> contextualize;
	contextualize --> route_question;
	grade_kb_evidence -.-> generate_from_kb;
	grade_kb_evidence -.-> search_web;
	grade_web_evidence -.-> answer_insufficient;
	grade_web_evidence -.-> generate_from_web;
	grade_web_evidence -.-> rewrite_query;
	retrieve_kb --> grade_kb_evidence;
	rewrite_query --> retrieve_kb;
	route_question -.-> direct_answer;
	route_question -.-> retrieve_kb;
	search_web --> grade_web_evidence;
	answer_insufficient --> __end__;
	direct_answer --> __end__;
	generate_from_kb --> __end__;
	generate_from_web --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
```
