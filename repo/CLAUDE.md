# Project instructions for the agent

Project: belief-driven multi-UAV victim search with MAPPO (B.Tech final-year project).
Read docs/BRIEF.md first. It is the briefing: project, code, results, known bugs, nearest papers, and a prioritised task list (Section 13).

## Rules
- Simulation first. Ignore hardware (lab drone, Raspberry Pi) until I say otherwise.
- Edit and test code in this repo with pytest. Use Colab only for long runs.
- Launch long training in the background (nohup) with logs and checkpoints on Google Drive. Never hold a notebook cell open for hours.
- After any run, confirm the checkpoint and log files exist on Drive before saying it is done.
- Never report a result that you did not read from a log file or test output.
- For every claim about a paper, give a link and a section pointer. Mark anything unverified.
- Write a failing test before fixing any bug.
- Commit after each phase. Update docs/STATUS.md (done, next, open questions).
- Do not edit docs/BRIEF.md. Propose changes in docs/STATUS.md instead.
- Explain what you changed and why in plain words at the end of each phase. I must be able to answer panel questions about this code.

## Colab
- The Colab MCP server is connected. Ask me to open a Colab notebook in the browser first, then open the browser connection.
- Secrets go in Colab Secrets, never in code or prompts.

## Layout
- sar_marl/      code: environment, MAPPO, training, evaluation, visualisation
- docs/          brief, status log, Review-1 report, papers, earlier outputs
