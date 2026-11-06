from urllib import response

INSTRUCTIONS = """
Your task is to answer questions from the course participants
based on the provided context.

Use the context to find relevant information and provide accurate
answers. If the answer is not found in the context,
respond with "I don't know."
"""


USER_PROMPT_TEMPLATE = """
Question:
{question}

Context:
{context}
"""


class RAGBase:
    def __init__(
        self,
        index,
        llm_client,
        instructions=INSTRUCTIONS,
        prompt_template=USER_PROMPT_TEMPLATE,
        course='llm-zoomcamp',
        model="claude-haiku-5-5"
    ):
        self.index = index
        self.llm_client = llm_client
        self.instructions = instructions
        self.course = course
        self.prompt_template = prompt_template
        self.model = model

    def search(self, query, num_results=5):
        boost_dict = {"question": 3.0, "section": 0.5}
        filter_dict = {"course": self.course}

        return self.index.search(
            query,
            num_results=num_results,
            boost_dict=boost_dict,
            filter_dict=filter_dict
        )

    def build_context(self, search_results):
        lines = []

        for doc in search_results:
            lines.append(doc["section"])
            lines.append("Q: " + doc["question"])
            lines.append("A: " + doc["answer"])
            lines.append("")

        return "\n".join(lines).strip()

    def build_prompt(self, query, search_results):
        context = self.build_context(search_results)
        return self.prompt_template.format(
            question=query, context=context
        )

    def llm(self, prompt):
        input_messages = [
            {"role": "user", "content": prompt}
        ]

        response = self.llm_client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=self.instructions,
            messages=input_messages
        )

        return "".join(block.text for block in response.content if block.type == "text")

    def rag(self, query):
        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        answer = self.llm(prompt)
        return answer






# def search(question, course='llm-zoomcamp'):
#             boost_dict = {"question": 2.0, "section": 0.5}
#             filter_dict = {"course": course}

#             return index.search(
#             question,
#             boost_dict=boost_dict,
#             filter_dict=filter_dict,
#             num_results=5)


# def build_context(search_results):
#     lines = []

#     for doc in search_results:
#         lines.append(doc["section"])
#         lines.append("Q: " + doc["question"])
#         lines.append("A: " + doc["answer"])
#         lines.append("")

#     return "\n".join(lines).strip()


# def build_prompt(question, search_results):
#     context = build_context(search_results)
#     prompt = USER_PROMPT_TEMPLATE.format(question=question, context=context)
#     return prompt.strip()


# def llm(instructions, user_prompt, model="claude-haiku-5-5"):
#     message_history = [
#         {"role": "user", "content": user_prompt}
#     ]

#     response = client.messages.create(
#         model=model,
#         max_tokens=1024,
#         system=instructions,
#         messages=message_history,
#     )

#     return "".join(block.text for block in response.content if block.type == "text")


# def llm(self, prompt):
#         input_messages = [
#             {"role": "user", "content": prompt}
#         ]

#         response = self.llm_client.responses.create(
#             model=self.model,
#             max_tokens=1024,
#             system=self.instructions,
#             input=input_messages
#         )

#         return "".join(block.text for block in response.content if block.type == "text")

