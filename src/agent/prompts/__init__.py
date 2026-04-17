main_agent_system_prompt = ""

with open("./src/agent/prompts/main_agent.md", "r") as f:
    main_agent_system_prompt = f.read()
