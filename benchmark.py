import streamlit as st
import pandas as pd
import altair as alt
import requests
from bs4 import BeautifulSoup

# Fetch ground truth answer from Wikipedia
def fetch_ground_truth(query):
    try:
        # Construct the Wikipedia search URL
        search_url = f"https://en.wikipedia.org/w/index.php?search={query.replace(' ', '+')}"
        
        # Make a GET request to the search URL
        response = requests.get(search_url)
        response.raise_for_status()  # Raise an exception for failed requests
        
        # Parse the HTML content of the search results page
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Find the first search result link on the page
        first_result_link = soup.find("div", class_="mw-search-result-heading").a["href"]
        
        # Construct the URL for the first search result (Wikipedia article)
        article_url = f"https://en.wikipedia.org{first_result_link}"
        
        # Make a GET request to the article URL
        article_response = requests.get(article_url)
        article_response.raise_for_status()  # Raise an exception for failed requests
        
        # Parse the HTML content of the article page
        article_soup = BeautifulSoup(article_response.content, "html.parser")
        
        # Find the first paragraph of the article (assumed to contain the ground truth answer)
        first_paragraph = article_soup.find("div", class_="mw-parser-output").p.get_text()
        
        return first_paragraph.strip()
    
    except requests.exceptions.RequestException as e:
        st.warning("Failed to fetch the ground truth answer from Wikipedia. Please check your internet connection and try again.")
        st.warning(str(e))
        return ""
    except Exception as e:
        st.warning("An error occurred while fetching the ground truth answer from Wikipedia.")
        st.warning(str(e))
        return ""

def main():
    st.title("AI Benchmarking with Ground Truth from Wikipedia")

    # Input the query
    st.subheader("Enter the query:")
    query = st.text_input("Query")

    # Fetch the ground truth answer from Wikipedia
    ground_truth_answer = fetch_ground_truth(query)

    # Display the Ground Truth answer
    st.subheader("Ground Truth from Wikipedia:")
    st.write(ground_truth_answer)

    # Calculate accuracy and relevance score
    def calculate_scores(ground_truth, ai_response):
        accuracy = 100 if ground_truth == ai_response else 0

        relevance_scores = [1]  # Simulated relevance scores (you can replace this with actual relevance scores)
        relevance_score = sum(relevance_scores) / len(relevance_scores)

        return accuracy, relevance_score

    def get_recommendation(accuracy_scores):
        max_accuracy = max(accuracy_scores)
        max_accuracy_idx = accuracy_scores.index(max_accuracy)

        return max_accuracy_idx

    # User Input for AI responses
    st.subheader("AI Responses:")
    st.write("Enter responses for each AI model:")
    chatgpt_response = st.text_input("ChatGPT Response")
    google_bart_response = st.text_input("Google BART Response")
    bing_ai_response = st.text_input("Bing AI Response")

    # Perform Benchmark button
    if st.button("Perform Benchmark"):
        # Calculate scores for each AI model
        chatgpt_accuracy, chatgpt_relevance_score = calculate_scores(ground_truth_answer, chatgpt_response)
        google_bart_accuracy, google_bart_relevance_score = calculate_scores(ground_truth_answer, google_bart_response)
        bing_ai_accuracy, bing_ai_relevance_score = calculate_scores(ground_truth_answer, bing_ai_response)

        # Display results
        st.subheader("AI Responses:")
        st.write(pd.DataFrame({"Query": [query], "ChatGPT": [chatgpt_response], "Google BART": [google_bart_response], "Bing AI": [bing_ai_response]}))

        st.subheader("Benchmark Results:")
        st.write("ChatGPT:")
        st.write(f"Accuracy: {chatgpt_accuracy:.2f}%")
        st.write(f"Relevance Score: {chatgpt_relevance_score:.2f}")
        st.write("Google BART:")
        st.write(f"Accuracy: {google_bart_accuracy:.2f}%")
        st.write(f"Relevance Score: {google_bart_relevance_score:.2f}")
        st.write("Bing AI:")
        st.write(f"Accuracy: {bing_ai_accuracy:.2f}%")
        st.write(f"Relevance Score: {bing_ai_relevance_score:.2f}")

        # Recommendations
        st.subheader("Recommendation:")
        accuracy_scores = [chatgpt_accuracy, google_bart_accuracy, bing_ai_accuracy]
        model_names = ["ChatGPT", "Google BART", "Bing AI"]
        max_accuracy_idx = get_recommendation(accuracy_scores)
        recommendation = model_names[max_accuracy_idx]
        st.write(f"Based on the benchmark, {recommendation} has the highest accuracy and is recommended for your use case.")

        # Bar chart visualization
        df = pd.DataFrame({"AI Model": model_names, "Accuracy": accuracy_scores})
        chart = alt.Chart(df).mark_bar().encode(
            x='AI Model',
            y='Accuracy',
            color=alt.condition(
                alt.datum.Accuracy == max(accuracy_scores),
                alt.value('green'),  # Highlight the highest accuracy bar
                alt.value('gray')
            )
        ).properties(width=500)
        st.subheader("Accuracy Comparison:")
        st.altair_chart(chart)

if __name__ == "__main__":
    main()
