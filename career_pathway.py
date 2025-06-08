import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import random
import sys
import os

# Check for required libraries
try:
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except ImportError:
    print("Error: NLTK is not installed. Run 'pip install nltk'")
    sys.exit(1)

# --- Mock Data for Job Listings and Coursera Courses ---
# Simulated job listings (matches your output)
mock_job_listings = [
    {
        "title": "Machine Learning Engineer",
        "skills": ["Python", "TensorFlow", "Deep Learning", "Data Analysis", "SQL", "AWS"],
        "description": "Develop and deploy machine learning models using Python and TensorFlow."
    },
    {
        "title": "Data Scientist",
        "skills": ["Python", "R", "Machine Learning", "Statistics", "Pandas", "Visualization"],
        "description": "Analyze datasets and build predictive models."
    },
    {
        "title": "AI Product Manager",
        "skills": ["AI Strategy", "Product Management", "Python", "Data Analysis", "Agile"],
        "description": "Lead AI product development with Python and Agile."
    },
    {
        "title": "Software Engineer",
        "skills": ["Java", "Python", "JavaScript", "SQL", "Git", "Agile"],
        "description": "Build scalable applications using Java and Python."
    },
    {
        "title": "Cloud Architect",
        "skills": ["AWS", "Azure", "Kubernetes", "Docker", "Python", "Cloud Security"],
        "description": "Design cloud infrastructure with AWS and Kubernetes."
    },
    {
        "title": "Data Analyst",
        "skills": ["SQL", "Tableau", "Python", "Excel", "Data Visualization", "Statistics"],
        "description": "Analyze data and create dashboards using SQL and Tableau."
    }
]

# Coursera courses (matches your output)
mock_coursera_courses = [
    {
        "title": "Machine Learning by Stanford University",
        "skills": ["Machine Learning", "Python", "Statistics"],
        "url": "https://www.coursera.org/learn/machine-learning",
        "description": "Learn the foundations of Machine Learning with Python."
    },
    {
        "title": "Deep Learning Specialization",
        "skills": ["Deep Learning", "TensorFlow", "Neural Networks"],
        "url": "https://www.coursera.org/specializations/deep-learning",
        "description": "Master deep learning techniques using TensorFlow."
    },
    {
        "title": "IBM AI Developer Professional Certificate",
        "skills": ["Python", "AI Development", "Data Analysis", "AI Strategy"],
        "url": "https://www.coursera.org/professional-certificates/ibm-ai-developer",
        "description": "Build AI-powered applications with Python."
    },
    {
        "title": "Data Visualization with Python",
        "skills": ["Visualization", "Python", "Pandas"],
        "url": "https://www.coursera.org/learn/python-for-data-visualization",
        "description": "Create visualizations using Python and Pandas."
    },
    {
        "title": "R Programming by Johns Hopkins",
        "skills": ["R", "Data Analysis", "Statistics"],
        "url": "https://www.coursera.org/learn/r-programming",
        "description": "Learn data analysis with R."
    },
    {
        "title": "Java Programming and Software Engineering",
        "skills": ["Java", "Software Development", "Git"],
        "url": "https://www.coursera.org/specializations/java-programming",
        "description": "Master Java programming and software engineering."
    },
    {
        "title": "AWS Cloud Solutions Architect",
        "skills": ["AWS", "Cloud Computing", "Kubernetes"],
        "url": "https://www.coursera.org/professional-certificates/aws-cloud-solutions-architect",
        "description": "Design cloud solutions with AWS."
    },
    {
        "title": "Data Analysis with Tableau",
        "skills": ["Tableau", "Data Visualization", "SQL"],
        "url": "https://www.coursera.org/learn/data-analysis-tableau",
        "description": "Create dashboards with Tableau."
    }
]

# --- Industry Trend Analyzer ---
class IndustryTrendAnalyzer:
    def __init__(self):
        """Initialize with mock job data (replace with real API in production)."""
        self.job_data = mock_job_listings

    def scrape_job_market(self, job_title_query="data scientist"):
        """
        Simulate scraping job market data for a given job title.
        Args:
            job_title_query (str): Job title to search for (e.g., 'Data Scientist').
        Returns:
            list: List of job dictionaries with titles, skills, and descriptions.
        """
        print(f"Scraping job market for: {job_title_query}")
        return self.job_data

    def extract_trending_skills(self, job_data):
        """
        Extract and rank skills by frequency across job listings.
        Args:
            job_data (list): List of job dictionaries.
        Returns:
            list: Sorted list of tuples (skill, frequency).
        """
        all_skills = []
        for job in job_data:
            all_skills.extend(job["skills"])
        
        # Count skill occurrences
        skill_freq = {}
        for skill in all_skills:
            skill_freq[skill] = skill_freq.get(skill, 0) + 1
        
        # Sort skills by frequency (descending)
        return sorted(skill_freq.items(), key=lambda x: x[1], reverse=True)

# --- Gap Detection Agent ---
class GapDetectionAgent:
    def __init__(self):
        """Initialize with NLTK stopwords for text preprocessing."""
        self.stop_words = set(stopwords.words('english'))
    
    def preprocess_text(self, text):
        """
        Preprocess text for NLP by tokenizing, removing stopwords, and normalizing.
        Args:
            text (str): Input text to preprocess.
        Returns:
            str: Cleaned and tokenized text, or placeholder if empty.
        """
        tokens = word_tokenize(text.lower())
        tokens = [t for t in tokens if t not in self.stop_words and t.isalnum()]
        return " ".join(tokens) if tokens else "placeholder"

    def compare_skills(self, user_skills, job_skills):
        """
        Compare user skills with job skills using TF-IDF and cosine similarity.
        Args:
            user_skills (list): List of user skills.
            job_skills (list): List of job-required skills.
        Returns:
            tuple: (similarity score, list of missing skills).
        """
        # Handle empty skill lists
        user_skills = user_skills or ["unknown"]
        job_skills = job_skills or ["unknown"]
        
        # Combine skills into text
        user_text = " ".join(user_skills)
        job_text = " ".join(job_skills)
        
        # Preprocess texts
        user_text = self.preprocess_text(user_text)
        job_text = self.preprocess_text(job_text)
        
        try:
            # Vectorize using TF-IDF
            vectorizer = TfidfVectorizer()
            vectors = vectorizer.fit_transform([user_text, job_text])
            
            # Calculate cosine similarity
            similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
            
            # Boost similarity for common skills (increased boost for better scores)
            common_skills = len(set(user_skills).intersection(set(job_skills)))
            similarity = min(similarity + (common_skills * 0.15), 1.0)  # Increased from 0.1 to 0.15
        except ValueError:
            # Fallback if vectorization fails
            similarity = 0.0
        
        # Identify missing skills (case-insensitive)
        missing_skills = [skill for skill in job_skills if skill.lower() not in [s.lower() for s in user_skills]]
        return similarity, missing_skills

# --- Micro-Credential Advisor ---
class MicroCredentialAdvisor:
    def __init__(self):
        """Initialize with mock Coursera course data."""
        self.courses = mock_coursera_courses
    
    def scrape_coursera(self, skill_query):
        """
        Simulate scraping Coursera for courses matching a skill.
        Args:
            skill_query (str): Skill to search for.
        Returns:
            list: List of course dictionaries matching the skill.
        """
        print(f"Scraping Coursera for courses related to: {skill_query}")
        matching_courses = [
            course for course in self.courses
            if skill_query.lower() in [s.lower() for s in course["skills"]]
        ]
        return matching_courses
    
    def recommend_courses(self, missing_skills):
        """
        Recommend Coursera courses to address missing skills, avoiding duplicates.
        Args:
            missing_skills (list): Skills the user needs to acquire.
        Returns:
            list: List of unique course recommendations with details.
        """
        recommendations = []
        seen_courses = set()  # Track unique course titles
        for skill in missing_skills:
            courses = self.scrape_coursera(skill)
            for course in courses:
                if course["title"] not in seen_courses:
                    recommendations.append({
                        "skill": skill,
                        "course_title": course["title"],
                        "url": course["url"],
                        "description": course["description"]
                    })
                    seen_courses.add(course["title"])
        return recommendations

# --- Career Topology Navigator ---
class CareerPathwayNavigator:
    def __init__(self, user_profiles):
        """
        Initialize with user profiles and component classes.
        Args:
            user_profiles (list): List of user profile dictionaries.
        """
        self.trend_analyzer = IndustryTrendAnalyzer()
        self.gap_detector = GapDetectionAgent()
        self.credential_advisor = MicroCredentialAdvisor()
        self.user_profiles = user_profiles
    
    def analyze_career_path(self, user_profile):
        """
        Analyze career path for a single user and provide recommendations.
        Args:
            user_profile (dict): User profile with name, skills, and career goal.
        Returns:
            dict: Analysis results including skill gaps and course recommendations.
        """
        try:
            # Extract user information
            user_skills = user_profile.get("skills", [])
            career_goal = user_profile.get("career_goal", "")
            user_name = user_profile.get("name", "Unknown User")
            
            # Step 1: Analyze job market for the career goal
            job_data = self.trend_analyzer.scrape_job_market(career_goal)
            trending_skills = self.trend_analyzer.extract_trending_skills(job_data)
            
            # Step 2: Find job matching the career goal
            target_job = next((job for job in job_data if job["title"].lower() == career_goal.lower()), None)
            if not target_job:
                return {"error": f"No job data found for {career_goal}"}
            
            # Step 3: Detect skill gaps
            similarity, missing_skills = self.gap_detector.compare_skills(user_skills, target_job["skills"])
            
            # Step 4: Recommend courses for missing skills
            course_recommendations = self.credential_advisor.recommend_courses(missing_skills)
            
            # Step 5: Compile results
            result = {
                "user": user_name,
                "career_goal": career_goal,
                "current_skills": user_skills,
                "trending_skills": trending_skills,
                "skill_match_score": similarity,
                "missing_skills": missing_skills,
                "course_recommendations": course_recommendations
            }
            return result
        except Exception as e:
            return {"error": f"Error processing {user_name}: {str(e)}"}

# --- Example Usage ---
def main():
    # Load user profiles from user_profiles.json
    try:
        if not os.path.exists("user_profiles.json"):
            print("Error: user_profiles.json not found. Creating a default file.")
            # Create default profiles if file doesn't exist
            default_profiles = [
                {
                    "name": "John Doe",
                    "skills": ["Python", "Data Analysis"],
                    "career_goal": "Data Scientist"
                },
                {
                    "name": "Jane Roe",
                    "skills": ["Python", "TensorFlow", "SQL"],
                    "career_goal": "Machine Learning Engineer"
                },
                {
                    "name": "Alex Kim",
                    "skills": ["Python", "Agile", "Product Management"],
                    "career_goal": "AI Product Manager"
                },
                {
                    "name": "Sofia Patel",
                    "skills": ["Java", "SQL", "Git"],
                    "career_goal": "Software Engineer"
                },
                {
                    "name": "Liam Chen",
                    "skills": ["AWS", "Python", "Docker"],
                    "career_goal": "Cloud Architect"
                },
                {
                    "name": "Emma Brown",
                    "skills": ["SQL", "Excel", "Tableau"],
                    "career_goal": "Data Analyst"
                },
                {
                    "name": "Noah Lee",
                    "skills": ["Python", "R", "Statistics"],
                    "career_goal": "Data Scientist"
                },
                {
                    "name": "Olivia Garcia",
                    "skills": ["JavaScript", "Python", "Agile"],
                    "career_goal": "Software Engineer"
                },
                {
                    "name": "Mason Wong",
                    "skills": ["Azure", "Kubernetes", "Python"],
                    "career_goal": "Cloud Architect"
                },
                {
                    "name": "Ava Martinez",
                    "skills": ["Python", "Data Visualization", "SQL"],
                    "career_goal": "Data Analyst"
                }
            ]
            with open("user_profiles.json", "w") as f:
                json.dump(default_profiles, f, indent=4)
        
        with open("user_profiles.json", "r") as f:
            user_profiles = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in user_profiles.json: {e}")
        return
    except Exception as e:
        print(f"Error loading user_profiles.json: {e}")
        return

    # Initialize navigator
    navigator = CareerPathwayNavigator(user_profiles)
    
    # Store all results
    all_results = []
    
    # Process each user profile
    for user_profile in user_profiles:
        result = navigator.analyze_career_path(user_profile)
        
        # Skip if error occurs
        if "error" in result:
            print(result["error"])
            continue
        
        # Append to results list
        all_results.append(result)
        
        # Display results
        print(f"\n{'='*50}")
        print(f"Career Pathway Analysis for {result['user']}")
        print(f"Career Goal: {result['career_goal']}")
        print(f"Current Skills: {', '.join(result['current_skills'])}")
        print(f"Skill Match Score: {result['skill_match_score']:.2%}")
        print("\nTrending Skills in Industry:")
        for skill, count in result['trending_skills']:
            print(f"- {skill}: {count} job listings")
        print("\nMissing Skills:")
        for skill in result["missing_skills"]:
            print(f"- {skill}")
        print("\nRecommended Courses:")
        for rec in result["course_recommendations"]:
            print(f"- Skill: {rec['skill']}")
            print(f"  Course: {rec['course_title']}")
            print(f"  URL: {rec['url']}")
            print(f"  Description: {rec['description']}\n")

    # Save results to career_pathways.json
    try:
        with open("career_pathways.json", "w") as f:
            json.dump(all_results, f, indent=4)
        print("\nResults saved to career_pathways.json")
    except Exception as e:
        print(f"Error saving career_pathways.json: {e}")

if __name__ == "__main__":
    # Check dependencies
    required_libs = ['requests', 'bs4', 'pandas', 'sklearn', 'nltk']
    for lib in required_libs:
        try:
            __import__(lib)
        except ImportError:
            print(f"Error: {lib} is not installed. Run 'pip install {lib}'")
            sys.exit(1)
    
    main()