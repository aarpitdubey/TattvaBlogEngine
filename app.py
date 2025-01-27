import os
import requests
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# Streamlit app configuration
st.set_page_config(page_title="Tattva Blog Engine")
st.title("Tattva Blog Engine")

# Load environment variables
_ = load_dotenv(find_dotenv())
OpenAI_API_KEY = os.getenv("OPENAI_API_KEY")
UNSPLASH_API_KEY = os.getenv("UNSPLASH_API_KEY")

# Check if API keys are set
if not OpenAI_API_KEY:
    st.warning("OpenAI API Key is missing. Please set it in your environment.")
if not UNSPLASH_API_KEY:
    st.warning("Unsplash API Key is missing. Please set it in your environment.")


def fetch_image_from_unsplash(query, api_key):
    """
    Fetch an image URL from Unsplash based on the query.

    Args:
        query (str): Search query for the image.
        api_key (str): Unsplash API key.

    Returns:
        str or None: The URL of the first image if found, otherwise None.
    """
    url = "https://api.unsplash.com/search/photos"
    params = {
        "query": query,
        "client_id": api_key,
        "per_page": 1,  # Limit results to 1 image
    }

    try:
        image_api_response = requests.get(url, params=params)
        image_api_response.raise_for_status()  # Raise HTTPError for bad responses
        data = image_api_response.json()
        if data.get("results"):
            return data["results"][0]["urls"]["regular"]
        else:
            st.info(f"No images found for query: {query}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching image for '{query}': {e}")
        return None


def blog_generation(topic):
    """
    Generate a blog based on a given topic, including fetching related images.

    Args:
        topic (str): The topic for the blog.
    """
    if not topic.strip():
        st.error("Please enter a valid topic to generate a blog.")
        return

    # Initialize the LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, openai_api_key=OpenAI_API_KEY)

    # Define the system prompt
    SYSTEM_TEMPLATE = """
    As an experienced writer, generate a 500-words blog post about {topic}.
    Include references to relevant images (e.g., maps, artifacts) by their names.
    paragraph or sub-paragraphs should be in heading styles starded from h3 to h4 headings.
    Format response as: 'Content'(don't show content as title but generate a perfect professional title for the blog post with h1 heading), 
    followed by 'Image Suggestions'(don't show these suggestions on blog post as content but use it for dusplaying the images) with image descriptions.
    Then, sum the total number of words in it and print the result like this: This post has X words.
    """

    # Create the prompt
    PROMPT_TEMPLATE = PromptTemplate(input_variables=["topic"], template=SYSTEM_TEMPLATE)
    prompt_query = PROMPT_TEMPLATE.format(topic=topic)

    # Get the response from OpenAI
    try:
        # Get the response from OpenAI
        llm_response = llm.predict(prompt_query).strip()
        # st.write("Raw LLM Response:", llm_response)  # Debugging: Display raw response

        if not llm_response:
            st.error("Empty response from the model.")
            return

        # Process content and image suggestions
        if "Image Suggestions:" in llm_response:
            content, image_suggestions = llm_response.split("Image Suggestions:", 1)
            image_suggestions = [line.strip() for line in image_suggestions.strip().split("\n") if line.strip()]
        else:
            content = llm_response.strip()
            image_suggestions = []  # No suggestions provided

        # Debugging: Display extracted content and image suggestions
        # st.subheader("Generated Blog Post:")
        st.write(content.strip())
        # st.write("Extracted Image Suggestions:", image_suggestions)  # Debugging

        # Fallback mechanism if no suggestions are provided
        if not image_suggestions:
            # st.info("No image suggestions provided by the model. Generating default suggestions.")
            image_suggestions = [f"{topic}: A generic visual representation"]

        # Display images with descriptions
        if image_suggestions:
            # st.subheader("Suggested Images for the Blog Post:")
            for line in image_suggestions:
                if ":" in line:
                    image_name, description = line.split(":", 1)
                    image_url = fetch_image_from_unsplash(image_name.strip(), UNSPLASH_API_KEY)
                    if image_url:
                        st.image(image_url, caption=description.strip(), use_container_width=True)
                    else:
                        st.warning(f"Could not fetch an image for: {image_name.strip()}")
        else:
            st.info("No image suggestions available.")

    except Exception as e:
        st.error(f"Error generating the blog post: {str(e)}")



# Streamlit UI
topic_input = st.text_input("Enter topic:")
if topic_input:
    with st.spinner("Generating your blog post..."):
        blog_generation(topic_input)
