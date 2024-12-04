import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from functools import lru_cache

def main(user_id, model_path="node2vec_model", user_data_path="user.json", threshold=0.2): # threshold for user-interested jobs skills matching

    node2vec_model = Word2Vec.load(model_path)
    
    with open(user_data_path, 'r') as file:
        users_data = json.load(file)

    @lru_cache(maxsize=1000)
    def get_skill_embedding(skill):
        try:
            return tuple(node2vec_model.wv[skill.lower()])
        except KeyError:
            return tuple(np.zeros(node2vec_model.vector_size))

    def calculate_similarity(skill_set1, skill_set2):
        skill_set1 = [skill.lower() for skill in skill_set1]
        skill_set2 = [skill.lower() for skill in skill_set2]

        embeddings1 = np.array([np.array(get_skill_embedding(skill)) for skill in skill_set1])
        embeddings2 = np.array([np.array(get_skill_embedding(skill)) for skill in skill_set2])
 
        if len(embeddings1) == 0 or len(embeddings2) == 0:
            return 0.0

        similarity_matrix = cosine_similarity(embeddings1, embeddings2)
        return float(np.mean(similarity_matrix))

    user = next((u for u in users_data if u["user"]["$oid"] == user_id), None)
    if not user:
        return f"User with ID {user_id} not found."

    user_skills = [skill["value"] for skill in user["skills"]]
    job_scores = []
    all_interested_skills = set()

    for job in user["jobs"]:
        job_skills = [skill["value"] for skill in job["jobSkills"]]
        similarity_score = calculate_similarity(user_skills, job_skills)
        
        if similarity_score >= threshold:
            job_scores.append({
                "jobTitle": job["jobTitle"], 
                "jobSkills": job_skills, 
                "similarityScore": similarity_score
            })
            all_interested_skills.update(job_skills)

    return {
        "userSkills": user_skills,
        "jobScores": job_scores,
        "allInterestedSkills": list(all_interested_skills)
    }

def match_user_to_jobs(user_id, model_path, all_interested_skills, employer_path, similarity_threshold=0.5, exact_match_weight=1.2): # threshold for interested jobs- all jobs 

    node2vec_model = Word2Vec.load(model_path)
    
    with open(employer_path, 'r') as employer_file:
        employers_data = json.load(employer_file)

    @lru_cache(maxsize=1000)
    def get_skill_embedding(skill):
        try:
            return tuple(node2vec_model.wv[skill.lower()])
        except KeyError:
            return tuple(np.zeros(node2vec_model.vector_size))

    def calculate_similarity(skill_set1, skill_set2):
        skill_set1 = [skill.lower() for skill in skill_set1]
        skill_set2 = [skill.lower() for skill in skill_set2]
        
        embeddings1 = np.array([np.array(get_skill_embedding(skill)) for skill in skill_set1])
        embeddings2 = np.array([np.array(get_skill_embedding(skill)) for skill in skill_set2])

        if len(embeddings1) == 0 or len(embeddings2) == 0:
            return 0.0

        similarity_matrix = cosine_similarity(embeddings1, embeddings2)
        base_similarity = float(np.mean(similarity_matrix))

        exact_matches = set(skill_set1) & set(skill_set2)
        weighted_similarity = base_similarity + len(exact_matches) * exact_match_weight

        length_weight = 2 * len(skill_set1) * len(skill_set2) / (len(skill_set1) + len(skill_set2))
        return (weighted_similarity * length_weight) / (len(skill_set1) + len(skill_set2))

    user_job_skills = [skill.lower() for skill in all_interested_skills]
    job_matches = []

    for employer in employers_data:
        job_title = employer.get("jobTitle", {}).get("value")
        employer_skills = [
            skill['value'].lower()
            for skill in employer.get("skills", [])
            if skill and "value" in skill
        ]

        if not job_title or not employer_skills:
            continue

        similarity_score = calculate_similarity(user_job_skills, employer_skills)

        if similarity_score > similarity_threshold:
            exact_matches = set(user_job_skills) & set(employer_skills)
            job_matches.append({
                "jobTitle": job_title,
                "employerSkills": employer_skills,
                "userJobSkills": user_job_skills,
                "similarityScore": similarity_score,
                "exactMatches": list(exact_matches)
            })

    job_matches.sort(key=lambda x: x['similarityScore'], reverse=True)

    return {
        "userJobSkills": user_job_skills,
        "matchedJobs": job_matches[:20]
    }

if __name__ == "__main__":
    USER_ID = "65f7cc7b6037d2f3adac728b"
    MODEL_PATH = "node2vec_model"
    USER_DATA_PATH = "user.json"
    EMPLOYER_PATH = "employer.json"

    result = main(USER_ID)
    
    if isinstance(result, dict):
        print(f"User Skills: {result['userSkills']}")
        for job in result["jobScores"]:
            print(f"Interested Job Title: {job['jobTitle']}")
            print(f"Interested Job Skills: {job['jobSkills']}")
            print(f"Similarity Score: {job['similarityScore']}")
        print(f"All Interested Skills: {result['allInterestedSkills']}")

        job_match_result = match_user_to_jobs(USER_ID, MODEL_PATH, result['allInterestedSkills'], EMPLOYER_PATH)
        
        print("\n--- Recommended Jobs ---")
        for job in job_match_result["matchedJobs"]:
            print(f"Job Title: {job['jobTitle']}")
            print(f"Employer Skills: {job['employerSkills']}")
            print(f"Exact Matches: {job['exactMatches']}")
            print(f"Similarity Score: {job['similarityScore']:.4f}\n")
    else:
        print(result)
        
# import json
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# from gensim.models import Word2Vec

# def main(user_id, model_path="node2vec_model", user_data_path="user.json", threshold=0.05):

#     # Load the Word2Vec model
#     node2vec_model = Word2Vec.load(model_path)

#     # Load user data
#     with open(user_data_path, 'r') as file:
#         users_data = json.load(file)

#     # Helper to get embeddings
#     def get_skill_embedding(skill):
#         try:
#             return node2vec_model.wv[skill]
#         except KeyError:
#             return np.zeros(node2vec_model.vector_size)

#     # Calculate similarity
#     def calculate_similarity(skill_set1, skill_set2):
#         skill_set1 = [skill.lower() for skill in skill_set1]
#         skill_set2 = [skill.lower() for skill in skill_set2]
#         embeddings1 = np.array([get_skill_embedding(skill) for skill in skill_set1])
#         embeddings2 = np.array([get_skill_embedding(skill) for skill in skill_set2])
#         similarity_matrix = cosine_similarity(embeddings1, embeddings2)
#         return np.mean(similarity_matrix)

#     # Find the user
#     user = next((u for u in users_data if u["user"]["$oid"] == user_id), None)
#     if not user:
#         return f"User with ID {user_id} not found."

#     user_skills = [skill["value"] for skill in user["skills"]]
#     job_scores = []
#     all_interested_skills = set()


#     for job in user["jobs"]:
#         job_skills = [skill["value"] for skill in job["jobSkills"]]
#         similarity_score = calculate_similarity(user_skills, job_skills)
        
#         if similarity_score >= threshold:
#             job_scores.append({"jobTitle": job["jobTitle"], "jobSkills": job_skills, "similarityScore": similarity_score})
#             all_interested_skills.update(job_skills)  # Add job skills to the merged set if above threshold

#     all_interested_skills = list(all_interested_skills)  # Convert the set to a list

#     return {
#         "userSkills": user_skills,
#         "jobScores": job_scores,
#         "allInterestedSkills": all_interested_skills
#     }
  
# # Example usage
# user_id = "665011a382fe1b556abb5467"
# result = main(user_id)

# if isinstance(result, dict):
#     print(f"User Skills: {result['userSkills']}")
#     for job in result["jobScores"]:
#         print(f"Interested Job Title: {job['jobTitle']}")
#         print(f"Interested Job Skills: {job['jobSkills']}")
#         print(f"Similarity Score: {job['similarityScore']}")
#     print(f"All Interested Skills: {result['allInterestedSkills']}")
# else:
#     print(result)
# User skills & interested job-skills match

# All interested job skills with all jobs match 
# import json
# import numpy as np
# import logging
# from sklearn.metrics.pairwise import cosine_similarity
# from gensim.models import Word2Vec

# # Configure logging
# logging.basicConfig(level=logging.DEBUG,
#                     format='%(asctime)s - %(levelname)s - %(message)s')

# class JobSkillsMatcher:
#     def __init__(self, model_path, user_path, employer_path):
#         """
#         Initialize JobSkillsMatcher with Word2Vec model and data files
#         """
#         # Load Word2Vec model
#         try:
#             self.node2vec_model = Word2Vec.load(model_path)
#             logging.info("Word2Vec model loaded successfully.")
#         except Exception as e:
#             logging.error(f"Failed to load Word2Vec model: {e}")
#             raise

#         # Load data files
#         try:
#             with open(user_path, 'r') as f:
#                 self.users_data = json.load(f)
#             with open(employer_path, 'r') as f:
#                 self.employers_data = json.load(f)
#             logging.info("User and employer data loaded successfully.")
#         except Exception as e:
#             logging.error(f"Failed to load data files: {e}")
#             raise

#     def get_skill_embedding(self, skill):
#         """
#         Get embedding for a skill, return zero vector if not found
#         """
#         try:
#             return self.node2vec_model.wv[skill.lower()]
#         except KeyError:
#             return np.zeros(self.node2vec_model.vector_size)

#     def calculate_similarity(self, skill_set1, skill_set2, exact_match_weight=1.2):
#         """
#         Calculate similarity between two skill sets, with additional weight for exact matches
#         and normalized by the length of the skill sets.
#         """
#         # Convert skills to lowercase
#         skill_set1 = [skill.lower() for skill in skill_set1]
#         skill_set2 = [skill.lower() for skill in skill_set2]

#         # Get embeddings for all skills
#         embeddings1 = np.array([self.get_skill_embedding(skill) for skill in skill_set1])
#         embeddings2 = np.array([self.get_skill_embedding(skill) for skill in skill_set2])

#         # Compute cosine similarity matrix
#         if len(embeddings1) == 0 or len(embeddings2) == 0:
#             return 0.0

#         similarity_matrix = cosine_similarity(embeddings1, embeddings2)

#         # Calculate exact matches
#         exact_matches = set(skill_set1) & set(skill_set2)

#         # Calculate base similarity as mean of the cosine similarity matrix
#         base_similarity = float(np.mean(similarity_matrix))

#         # Calculate weighted similarity
#         weighted_similarity = base_similarity + len(exact_matches) * exact_match_weight

#         # Normalize by skill set lengths to balance the impact of skill set sizes
#         length_weight = 2 * len(skill_set1) * len(skill_set2) / (len(skill_set1) + len(skill_set2))
#         normalized_similarity = (weighted_similarity * length_weight) / (len(skill_set1) + len(skill_set2))

#         return normalized_similarity

#     def match_job_skills(self, user_id, similarity_threshold=0.1):
#         """
#         Match user's job skills with employer skills based on similarity.
#         """
#         # Find the user from the dataset by user_id
#         user = next((u for u in self.users_data if u["user"]["$oid"] == user_id), None)
#         if not user:
#             logging.error(f"User with ID {user_id} not found.")
#             return None

#         # Extract job skills from the user's previous jobs
#         user_job_skills = [
#             skill['value'].lower() 
#             for job in user.get('jobs', []) 
#             for skill in job.get('jobSkills', [])
#         ]
#         logging.info(f"User's Job Skills: {user_job_skills}")

#         # List to store job matches
#         job_matches = []

#         for employer in self.employers_data:
#             # Extract job title and employer's skills
#             job_title = employer.get("jobTitle", {}).get("value")
#             employer_skills = [
#                 skill['value'].lower() 
#                 for skill in employer.get("skills", []) 
#                 if skill and "value" in skill
#             ]

#             if not job_title or not employer_skills:
#                 continue

#             # Calculate similarity score between user skills and employer job skills
#             similarity_score = self.calculate_similarity(user_job_skills, employer_skills)

#             # If similarity score is above threshold, consider this job match
#             if similarity_score > similarity_threshold:
#                 exact_matches = set(user_job_skills) & set(employer_skills)  # Exact matches
#                 job_matches.append({
#                     "jobTitle": job_title,
#                     "employerSkills": employer_skills,
#                     "userJobSkills": user_job_skills,
#                     "similarityScore": similarity_score,
#                     "exactMatches": list(exact_matches)  # Optional field for logging
#                 })

#         # Sort jobs by similarity score (descending)
#         job_matches.sort(key=lambda x: x['similarityScore'], reverse=True)

#         top_20_matches = job_matches[:20]

#         return {
#             "userJobSkills": user_job_skills,
#             "matchedJobs": top_20_matches
#         }

# def main():
#     # Paths to your model and data files
#     MODEL_PATH = "node2vec_model"  # Path to your Word2Vec model
#     USER_PATH = "user.json"  # Path to the user data JSON file
#     EMPLOYER_PATH = "employer.json"  # Path to the employer data JSON file

#     # User ID to match (replace with the desired user ID)
#     USER_ID = "65f7cc7b6037d2f3adac728b"  # Replace with the user ID you want to match

#     try:
#         # Initialize the matcher
#         matcher = JobSkillsMatcher(MODEL_PATH, USER_PATH, EMPLOYER_PATH)

#         # Get matching jobs for the user
#         results = matcher.match_job_skills(USER_ID)

#         if results and results['matchedJobs']:
#             print("\n--- Matching Jobs ---")
#             for job in results['matchedJobs']:
#                 print(f"Job Title: {job['jobTitle']}")
#                 print(f"Employer Skills: {job['employerSkills']}")
#                 print(f"Exact Matches: {job['exactMatches']}")
#                 print(f"Similarity Score: {job['similarityScore']:.4f}\n")
#         else:
#             print("No matching jobs found.")

#     except Exception as e:
#         logging.error(f"An error occurred: {e}")

# if __name__ == "__main__":
#     main()        

