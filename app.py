from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import os
import json
import re
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all domains

# Configure Gemini API
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY environment variable not set!")
    raise ValueError("GEMINI_API_KEY environment variable not set!")

genai.configure(api_key=GEMINI_API_KEY)

# Initialize Gemini model
try:
    model = genai.GenerativeModel('gemini-1.5-flash')
    logger.info("Gemini model initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Gemini model: {e}")
    raise


def create_fake_news_analysis_prompt(article_text):
    """Create a structured prompt for fake news analysis"""
    return f"""
You are an expert fact-checker and media analyst specializing in detecting fake news and misinformation. 
Analyze the following news article text for credibility, authenticity, and potential misinformation.

Please analyze these aspects:
1. Source credibility and attribution
2. Factual accuracy and verifiable claims
3. Emotional manipulation and sensational language
4. Missing context or supporting evidence
5. Logical consistency and coherence
6. Signs of propaganda or deliberate misinformation

Article Text:
{article_text}

Please provide your analysis in the following JSON format:
{{
    "credibility_score": [number from 0-10],
    "verdict": "[CREDIBLE/SUSPICIOUS/FAKE/MIXED]",
    "confidence": [number from 0-100],
    "analysis": "[detailed explanation of your findings]",
    "red_flags": "[main credibility concerns]",
    "credibility_factors": "[positive credibility indicators]",
    "verification_tips": "[suggestions for fact-checking]"
}}

IMPORTANT: Return ONLY the JSON object, no additional text before or after.
"""


def extract_json_from_response(response_text):
    """Extract JSON from Gemini response, handling various formats"""
    try:
        return json.loads(response_text.strip())
    except json.JSONDecodeError:
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # fallback
        return {
            "credibility_score": 0,
            "verdict": "UNKNOWN",
            "confidence": 50,
            "analysis": response_text.strip() if response_text.strip() else "Unable to analyze the article.",
            "red_flags": "Response parsing failed",
            "credibility_factors": "Unable to assess",
            "verification_tips": "Please try again with a different article"
        }


def validate_article_text(text):
    """Validate the input article text"""
    if not text or not text.strip():
        return False, "Article text cannot be empty"

    if len(text.strip()) < 10:
        return False, "Article text is too short for meaningful analysis"

    if len(text.strip()) > 50000:
        return False, "Article text is too long (maximum 50,000 characters)"

    return True, ""


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "Fake News Detector API"
    })


@app.route('/analyze', methods=['POST'])
def analyze_authenticity():
    """Main endpoint for fake news analysis"""
    try:
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400

        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        article_text = data.get('text', '').strip()

        # Validate
        is_valid, error_message = validate_article_text(article_text)
        if not is_valid:
            return jsonify({"error": error_message}), 400

        logger.info(f"Analyzing article of length: {len(article_text)} characters")

        # Create prompt
        prompt = create_fake_news_analysis_prompt(article_text)

        # Call Gemini API
        try:
            response = model.generate_content(prompt)

            if not response or not response.text:
                raise Exception("Empty response from Gemini API")

            logger.info("Received response from Gemini API")

        except Exception as api_error:
            logger.error(f"Gemini API error: {api_error}")
            return jsonify({
                "error": "AI analysis service is temporarily unavailable. Please try again later."
            }), 503

        # Extract JSON
        try:
            analysis_result = extract_json_from_response(response.text)
            result = {
                "credibility_score": min(10, max(0, int(analysis_result.get("credibility_score", 0)))),
                "verdict": analysis_result.get("verdict", "UNKNOWN"),
                "confidence": min(100, max(0, int(analysis_result.get("confidence", 50)))),
                "analysis": analysis_result.get("analysis", "Analysis completed"),
                "red_flags": analysis_result.get("red_flags", "No major red flags identified"),
                "credibility_factors": analysis_result.get("credibility_factors", "Standard credibility markers present"),
                "verification_tips": analysis_result.get("verification_tips", "Cross-check with multiple reliable sources")
            }

            logger.info(f"Analysis completed - Score: {result['credibility_score']}, Verdict: {result['verdict']}")
            return jsonify(result)

        except Exception as parse_error:
            logger.error(f"Response parsing error: {parse_error}")
            return jsonify({"error": "Failed to parse analysis results. Please try again."}), 500

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return jsonify({"error": "An unexpected error occurred. Please try again."}), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({"error": "Method not allowed"}), 405


@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({"error": "Internal server error"}), 500


if __name__ == '__main__':
    if not GEMINI_API_KEY:
        print("ERROR: GEMINI_API_KEY environment variable not set!")
        exit(1)

    print("Starting Fake News Detector API...")
    print("Health Check: http://localhost:5000/health")
    app.run(debug=True, host='0.0.0.0', port=5000)
