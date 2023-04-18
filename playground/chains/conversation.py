from langchain import OpenAI, ConversationChain

llm = OpenAI(temperature=0)
conversation = ConversationChain(llm=llm, verbose=True)

print(conversation.predict(input="Hi there!"))


print(conversation.predict(
    input="I'm doing well! Just having a conversation with an AI."))
