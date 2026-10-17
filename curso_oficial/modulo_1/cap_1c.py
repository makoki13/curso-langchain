#####Initialising and invoking an agent
from langchain.agents import create_agent  # noqa: E402
from langchain_groq import ChatGroq  # noqa: E402
from pprint import pprint  # noqa: E402


model = ChatGroq(
            model="qwen/qwen3.8-27b",
            temperature=0.0
        )

agent = create_agent(model=model)
from langchain.messages import HumanMessage  # noqa: E402
response = agent.invoke(
    {"messages": [HumanMessage(content="What's the capital of the Moon?")]}
)
pprint(response)
print(response['messages'][-1].content)
from langchain.messages import AIMessage  # noqa: E402
response = agent.invoke(
    {"messages": 
        [
            HumanMessage(content="What's the capital of the Moon?"),
            AIMessage(content="The capital of the Moon is Luna City."),
            HumanMessage(content="Interesting, tell me more about Luna City")
        ]
    }
)
pprint(response)

#####Streaming Output
for token, metadata in agent.stream(
    {
"messages": 
[HumanMessage(
content="Tell me all about Luna City, the capital of the Moon"
)
]
    },
    stream_mode="messages"
):
    # token is a message chunk with token content
    # metadata contains which node produced the token
    
    if token.content:  # type: ignore # Check if there's actual content
        print(token.content, end="", flush=True)  # type: ignore # Print token
