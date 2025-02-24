# views.py
from django.shortcuts import render
from django.http import JsonResponse
from rest_framework.decorators import api_view
from django.views.decorators.csrf import csrf_exempt
import requests
from bs4 import BeautifulSoup
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from fuzzywuzzy import process
from .models import ExtractedURL
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import phonenumbers
from transformers import pipeline  # For advanced summarization
import logging


# Load spaCy's pre-trained model
nlp = spacy.load("en_core_web_sm")

# Load Hugging Face summarization pipeline (optional, for better summarization)
summarizer = pipeline("summarization")

# Set up logging
logging.basicConfig(level=logging.DEBUG)


@csrf_exempt
def index(request):
    if request.method == 'POST':
        base_url = request.POST.get('url')
        if not base_url:
            return JsonResponse({
                'status': 'error',
                'message': 'URL is required'
            })

        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(base_url, headers=headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')
            urls = set()

            for link in soup.find_all('a', href=True):
                href = link['href'].strip()
                if href and not href.startswith(('#', 'javascript:', 'mailto:')):
                    if href.startswith('http'):
                        urls.add(href)
                    elif href.startswith('//'):
                        urls.add(f"https:{href}")
                    elif href.startswith('/'):
                        urls.add(requests.compat.urljoin(base_url, href))

            for url in urls:
                try:
                    url_response = requests.get(url, headers=headers, timeout=10)
                    url_response.raise_for_status()
                    url_soup = BeautifulSoup(url_response.content, 'html.parser')

                    # Remove irrelevant elements
                    for tag in url_soup(['script', 'style', 'nav', 'footer', 'aside', 'header']):
                        tag.decompose()

                    # Extract text from main content
                    main_content = url_soup.find('main') or url_soup.find('article') or url_soup.find('section')
                    if main_content:
                        extracted_text = main_content.get_text(separator=' ').strip()
                    else:
                        extracted_text = url_soup.get_text(separator=' ').strip()

                    extracted_text = re.sub(r'\s+', ' ', extracted_text).strip()

                    # Save to database
                    ExtractedURL.objects.create(url=url, base_url=base_url, content=extracted_text)
                except Exception as e:
                    logging.error(f"Failed to scrape {url}: {e}")
                    continue

            return JsonResponse({
                'status': 'success',
                'urls': list(urls)
            })

        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to fetch URL: {str(e)}")
            return JsonResponse({
                'status': 'error',
                'message': f'Failed to fetch URL: {str(e)}'
            })
        except Exception as e:
            logging.error(f"An unexpected error occurred: {str(e)}")
            return JsonResponse({
                'status': 'error',
                'message': f'An unexpected error occurred: {str(e)}'
            })

    return render(request, 'index.html')


def chat_view(request):
    return render(request, 'chatbot.html')


def find_best_match(query, choices):
    best_match = process.extractOne(query, choices)
    if best_match and best_match[1] > 50:  # Ensure a minimum similarity score
        return best_match
    return None


@api_view(['POST'])
@csrf_exempt
def chat(request):
    try:
        user_query = request.data.get('message')
        logging.debug(f"User Query: {user_query}")
    except Exception as e:
        logging.error(f"Invalid request data: {e}")
        return JsonResponse({
            'status': 'error',
            'message': 'Invalid request data'
        })

    if not user_query:
        return JsonResponse({
            'status': 'error',
            'message': 'Query is required'
        })

    # Fetch all content from the database
    urls = ExtractedURL.objects.all()
    contents = [url.content for url in urls if url.content and url.content.strip()]
    logging.debug(f"Scraped Contents: {contents}")

    if not contents:
        return JsonResponse({
            'status': 'error',
            'message': 'No valid content available to process'
        })

    # Detect user intent
    intent = detect_intent(user_query)
    logging.debug(f"Detected Intent: {intent}")

    # Handle specific intents
    if intent == "contact":
        response = handle_contact_intent(contents)
    elif intent == "about":
        response = handle_about_intent(contents)
    elif intent == "services":
        response = handle_services_intent(contents)
    elif intent == "team":
        response = handle_team_intent(contents)
    elif intent == "general":
        response = handle_general_query(user_query, contents)
    else:
        response = handle_fallback_query(user_query, contents)

    logging.debug(f"Bot Response: {response}")
    return JsonResponse({
        'status': 'success',
        'response': response
    })


def detect_intent(query):
    """
    Detect the user's intent using spaCy with improved logic.
    """
    doc = nlp(query.lower())
    intents = {
        "contact": ["contact", "email", "phone", "call", "reach", "number", "how to reach"],
        "about": ["about", "who", "what is", "describe", "explain", "company", "organization"],
        "services": ["services", "offer", "provide", "do", "what do you do", "capabilities"],
        "team": ["team", "staff", "people", "members", "who works here", "leadership"],
        "general": ["how", "where", "when", "why", "what"]
    }

    # Count keyword matches for each intent
    intent_scores = {intent: 0 for intent in intents}
    for token in doc:
        for intent, keywords in intents.items():
            if token.text in keywords:
                intent_scores[intent] += 1

    # Return the intent with the highest score
    best_intent = max(intent_scores, key=intent_scores.get)
    return best_intent if intent_scores[best_intent] > 0 else "unknown"


def handle_contact_intent(contents):
    """
    Handle queries related to contact information.
    """
    emails = set()
    phones = set()

    for content in contents:
        # Extract emails (improved regex)
        emails.update(re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', content))

        # Extract and validate phone numbers (handle international formats)
        for match in phonenumbers.PhoneNumberMatcher(content, None):  # None for international numbers
            try:
                parsed_number = phonenumbers.parse(match.raw_string, None)
                if phonenumbers.is_valid_number(parsed_number):
                    phones.add(phonenumbers.format_number(parsed_number, phonenumbers.PhoneNumberFormat.INTERNATIONAL))
            except phonenumbers.phonenumberutil.NumberParseException:
                continue

    if emails or phones:
        response = "You can contact them via:\n"
        if emails:
            response += "Email:\n"
            response += "\n".join([f"  {email}" for email in emails]) + "\n"

        if phones:
            response += "Phone:\n"
            response += "\n".join([f"  {phone}" for phone in phones]) + "\n"
        response += "\nFor more details, please refer to the website."
    else:
        response = "I couldn't find any contact information. Please check the website for more details."
    return response


def extract_keywords(text, top_n=10):
    """
    Extract top keywords from the text using TF-IDF.
    """
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.toarray()[0]
    top_keywords = [feature_names[i] for i in tfidf_scores.argsort()[-top_n:][::-1]]
    return top_keywords


def is_relevant(text):
    """
    Determine if the text is relevant based on extracted keywords.
    """
    irrelevant_keywords = {"privacy", "policy", "terms", "conditions", "disclaimer", "cookies"}
    keywords = set(extract_keywords(text))
    return not keywords.intersection(irrelevant_keywords)


def filter_irrelevant_content(content):
    """
    Filter out irrelevant content using dynamic keyword extraction.
    """
    if not is_relevant(content):
        return ""
    return content


def summarize_content(content_list):
    """
    Summarize a list of content and format it as bullet points.
    """
    combined_content = " ".join(content_list)
    try:
        # Use Hugging Face summarization
        summary = summarizer(combined_content, max_length=130, min_length=30, do_sample=False)
        summary_text = summary[0]['summary_text']
        # Format as bullet points
        summary_points = summary_text.split(". ")
        return "\n".join([f"- {point.strip()}" for point in summary_points if point.strip()])
    except Exception as e:
        logging.error(f"Advanced summarization failed, falling back to sumy: {e}")
        parser = PlaintextParser.from_string(combined_content, Tokenizer("english"))
        summarizer_lsa = LsaSummarizer()
        summary = summarizer_lsa(parser.document, 3)  # Summarize to 3 sentences
        summary_sentences = [str(sentence) for sentence in summary]
        return "\n".join([f"- {sentence.strip()}" for sentence in summary_sentences if sentence.strip()])


def handle_about_intent(contents):
    """
    Handle queries related to "about" information.
    """
    about_keywords = ["about", "who we are", "company", "organization", "mission"]
    relevant_content = []

    for content in contents:
        filtered_content = filter_irrelevant_content(content)
        if filtered_content and any(keyword in filtered_content.lower() for keyword in about_keywords):
            relevant_content.append(filtered_content)

    if relevant_content:
        return summarize_content(relevant_content)
    else:
        return "I couldn't find any information about the company. Please check the website for more details."


def handle_services_intent(contents):
    """
    Handle queries related to services offered.
    """
    services_keywords = ["service", "offer", "provide", "capability", "solution"]
    relevant_content = []

    for content in contents:
        filtered_content = filter_irrelevant_content(content)
        if filtered_content and any(keyword in filtered_content.lower() for keyword in services_keywords):
            relevant_content.append(filtered_content)

    if relevant_content:
        return summarize_content(relevant_content)
    else:
        return "I couldn't find any information about services. Please check the website for more details."


def handle_team_intent(contents):
    """
    Handle queries related to the team or staff.
    """
    team_keywords = ["team", "staff", "member", "leadership", "people"]
    relevant_content = []

    for content in contents:
        filtered_content = filter_irrelevant_content(content)
        if filtered_content and any(keyword in filtered_content.lower() for keyword in team_keywords):
            relevant_content.append(filtered_content)

    if relevant_content:
        return summarize_content(relevant_content)
    else:
        return "I couldn't find any information about the team. Please check the website for more details."


def handle_general_query(query, contents):
    """
    Handle general queries using TF-IDF and cosine similarity with preprocessing.
    """
    try:
        # Preprocess contents
        processed_contents = [
            re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', '', content)).strip().lower()
            for content in contents if content.strip()
        ]

        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(processed_contents + [query])
        cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
        best_match_index = cosine_similarities.argmax()
        best_match_score = cosine_similarities[0, best_match_index]

        if best_match_score > 0.6:  # Increased threshold
            best_match_content = processed_contents[best_match_index]
            try:
                # Use Hugging Face summarization if available
                summary = summarizer(best_match_content, max_length=130, min_length=30, do_sample=False)
                response = summary[0]['summary_text']
            except Exception as e:
                logging.error(f"Advanced summarization failed, falling back to sumy: {e}")
                parser = PlaintextParser.from_string(best_match_content, Tokenizer("english"))
                summarizer_lsa = LsaSummarizer()
                summary = summarizer_lsa(parser.document, 3)  # Summarize to 3 sentences
                summary_sentences = [str(sentence) for sentence in summary]
                response = " ".join(summary_sentences)
        else:
            response = "No relevant information found."

        return response
    except Exception as e:
        logging.error(f"Error during processing: {e}")
        return "An error occurred while processing your request."


def handle_fallback_query(query, contents):
    """
    Handle fallback queries with improved guidance.
    """
    return (
        "I'm sorry, I couldn't understand your question. "
    )