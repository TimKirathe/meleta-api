from typing import Any, Dict, List, Union

import openai
from openai import AsyncOpenAI, OpenAIError
from openai.types.chat import ChatCompletion

from models import ApiResponse, Verse


class Helper:
    @staticmethod
    async def fetch_relevant_verses_stream(
        openai_client: AsyncOpenAI, user_query: str, stream: bool = True
    ) -> Union[str, None]:
        messages = [
            {
                "role": "system",
                "content": """
                The user query reflects a topic they want to meditate on from the Bible, as if they said:
                "I want Bible verses about <query>." Return up to 25 KJV verses most relevant to the query.
                """,
            },
            {
                "role": "user",
                "content": user_query,
            },
        ]

        response = await Helper.get_openai_chat_completion(
            openai_client=openai_client, messages=messages, response_format=Verse
        )

        if response and stream:
            return response
        elif response:
            response_usage = response.usage
            print(f"Completion Tokens: {response_usage.completion_tokens}")
            print(f"Prompt Tokens: {response_usage.prompt_tokens}")
            return response.choices[0].message.content
        else:
            return None

    @staticmethod
    async def fetch_relevant_verses(
        openai_client: AsyncOpenAI, user_query: str
    ) -> Union[str, None]:
        messages = [
            {
                "role": "system",
                "content": """
                The user query reflects a topic they want to meditate on from the Bible, as if they said:
                "I want Bible verses about <query>." Return up to 20 KJV verses most relevant to the query.
                Respond with a valid JSON array of verse objects in format:
                {
                    "book (str)": "Book",
                    "chapter (int)": "Chapter",
                    "versesText (str)": "Full text",
                    "versesNumRange (str)": "Verse number(s) range"
                }
                The JSON array should always be directly parsable by python json.loads()
                Don't add any extra characters. Just the data I have told you.
                """,
            },
            {"role": "user", "content": user_query},
        ]
        response = await Helper.get_openai_chat_completion(
            openai_client, messages, Verse
        )

        if response:
            response_usage = response.usage
            print(f"Completion Tokens: {response_usage.completion_tokens}")
            print(f"Prompt Tokens: {response_usage.prompt_tokens}")
            return response.choices[0].message.content
        else:
            return None

    @staticmethod
    async def clean_query(
        openai_client: AsyncOpenAI, user_query: str
    ) -> Union[str, None]:
        messages = [
            {
                "role": "system",
                "content": """
                I want you to paraphrase the text very concisely. Remove stop and filler words.
                Max 5 words.
                """,
            },
            {"role": "user", "content": user_query},
        ]

        response = await Helper.get_openai_chat_completion(
            openai_client, messages, model="gpt-4-turbo"
        )
        if response:
            response_usage = response.usage
            print(f"Completion Tokens: {response_usage.completion_tokens}")
            print(f"Prompt Tokens: {response_usage.prompt_tokens}")
            return response.choices[0].message.content

        else:
            return None

    @staticmethod
    async def get_openai_chat_completion(
        openai_client: AsyncOpenAI,
        messages: List[Dict[str, str]],
        response_format: Any = None,
        model: str = "gpt-4o",
        stream: bool = False,
        **kwargs,
    ) -> Union[ChatCompletion, None]:
        try:
            if stream:
                response = await openai_client.beta.chat.completions.stream(
                    model=model, messages=messages, response_format=response_format
                )
            else:
                response = await openai_client.chat.completions.create(
                    model=model, messages=messages, **kwargs
                )
            return response
        except openai.RateLimitError as e:
            print("Rate limit hit:", e)
        except openai.AuthenticationError as e:
            print("Invalid API key:", e)
        except openai.InvalidRequestError as e:
            print("Bad request:", e)
        except openai.APIConnectionError as e:
            print("Failed to connect to API:", e)
        except openai.APIError as e:
            print("API Server Error:", e)
        except OpenAIError as e:
            print("Unknown OpenAI error:", e)

        return None

    @staticmethod
    def generate_api_response(
        success: bool, data: Union[int, str, dict, list, None], **kwargs
    ) -> ApiResponse:
        return ApiResponse(success=success, data=data, **kwargs)
