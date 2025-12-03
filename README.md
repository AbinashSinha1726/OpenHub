📘 CrewAI Research & Blog Automation

Automate research and content generation using CrewAI, OpenAI GPT-5.1, and Serper Search API.
This project creates a two-agent workflow:

. Research Agent → finds verified, recent facts about a chosen topic
. Writer Agent → generates a polished 100-word blog summary

Perfect for content creators, marketers, developers, and AI automation builders.

🚀 Features

🔍 Automated research using SerperDevTool
✍️ AI-generated blog writing using GPT-5.1
🤖 Multi-agent collaboration via CrewAI
📘 Simple two-task workflow: research → write
🔑 Environment-based API key handling

🛠️ Installation
1. Clone the repository
   git clone https://github.com/yourusername/your-repo.git
   cd your-repo
2. Create and activate a virtual environment
   python -m venv venv
   source venv/bin/activate      # macOS/Linux
   venv\Scripts\activate         # Windows
3. Install dependencies
   pip install -r requirements.txt

🔐 Environment Variables

Create a .env file in the root directory:

OPENAI_API_KEY=your_openai_key
SERPER_API_KEY=your_serper_key

Make sure you have active keys for:
OpenAI GPT-5.1
Serper Search API

📄 Usage

Run the script:
python main.py

You will see:
.Research logs
.Writer agent summary
.Final result printed in the console

📂 Project Structure
├── main.py
├── .env
├── requirements.txt
├── README.md

📘 Example Output
OPENAI API Key is Loaded: True
SERPER API Key is Loaded: True

--- Research Agent Output ---
- Fact 1...
- Fact 2...
- Fact 3...

--- Writer Agent Blog Summary ---
(100-word AI-crafted summary)

🤖 How It Works
🧠 1. Research Agent

Uses SerperDevTool() to gather:

Timely facts
Verified data
Recent insights

✍️ 2. Writer Agent

Transforms the research into:

A short blog
Around 100 words
Clear, engaging, and informative

🔄 Crew Flow

Task 1 runs → gathers facts
Task 2 reads the result of Task 1
Produces a polished blog summary

📦 Dependencies

crewai
crewai-tools
python-dotenv
openai
serper search integration
(Include versions in requirements.txt)

🧬 Customization

You can easily modify:

Number of agents
Writing style
Output format
Topic input
Task dependency chain

If you want, I can generate:
Multi-agent version (research → analysis → writing → editing)
API/CLI wrapper
FastAPI server version
Web UI version

🤝 Contributing

Pull requests are welcome!
For major changes, please open an issue first to discuss what you’d like to change.
