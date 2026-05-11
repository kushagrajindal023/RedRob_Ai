"""
Redrob AI Campus Hackathon — Resume Matching Engine
Language: Python (standard library only)
"""
import math

# ─────────────────────────────────────────────
# SKILL ALIASES  (exact as provided — do NOT modify)
# ─────────────────────────────────────────────
SKILL_ALIASES = {
    "python": "python", "pyhton": "python",
    "java": "java",
    "javascript": "javascript", "javascrpit": "javascript", "js": "javascript",
    "typescript": "typescript", "typescrpit": "typescript",
    "c++": "cpp", "cpp": "cpp",
    "r": "r", "kotlin": "kotlin",
    "machinelearning": "machine_learning", "machine learning": "machine_learning",
    "ml": "machine_learning", "sklearn": "machine_learning",
    "deeplearning": "deep_learning", "deep learning": "deep_learning",
    "deep-learning": "deep_learning",
    "tensorflow": "tensorflow", "pytorch": "pytorch", "keras": "keras",
    "nlp": "nlp", "bert": "bert", "xgboost": "xgboost",
    "feature engineering": "feature_engineering",
    "statistics": "statistics", "stats": "statistics",
    "regression": "regression", "clustering": "clustering",
    "data-viz": "data_visualization", "data visualization": "data_visualization",
    "data viz": "data_visualization", "matplotlib": "data_visualization",
    "tableau": "data_visualization", "power-bi": "data_visualization",
    "power bi": "data_visualization", "powerbi": "data_visualization",
    "pandas": "pandas", "numpy": "numpy",
    "react": "react", "reacts": "react", "reactjs": "react",
    "vue": "vue", "vue.js": "vue", "vuejs": "vue",
    "redux": "redux", "tailwind": "tailwind",
    "html/css": "html_css", "html css": "html_css",
    "html": "html_css", "css": "html_css",
    "jest": "jest", "graphql": "graphql",
    "node.js": "nodejs", "nodejs": "nodejs", "node js": "nodejs",
    "flask": "flask",
    "spring boot": "spring_boot", "springboot": "spring_boot",
    "rest api": "rest_api", "rest": "rest_api", "restapi": "rest_api",
    "microservices": "microservices",
    "sql": "sql", "mysql": "mysql", "mysq": "mysql",
    "postgresql": "postgresql", "postgres": "postgresql",
    "mongodb": "mongodb", "redis": "redis",
    "docker": "docker",
    "kubernetes": "kubernetes", "kubernates": "kubernetes", "k8s": "kubernetes",
    "ci/cd": "ci_cd", "cicd": "ci_cd", "ci cd": "ci_cd",
    "aws": "aws",
    "android": "android", "firebase": "firebase",
    "algorithms": "algorithms", "algoritms": "algorithms",
    "data structure": "data_structures", "data structures": "data_structures",
    "competitive programming": "competitive_programming",
    "ui/ux": "ui_ux", "ui ux": "ui_ux", "figma": "figma",
}

# Sort aliases: longer (multi-word) first so they are matched before sub-tokens
SORTED_ALIASES = sorted(SKILL_ALIASES.keys(), key=lambda x: len(x.split()), reverse=True)


# ─────────────────────────────────────────────
# STEP 1 & 2 — Normalize + Deduplicate
# ─────────────────────────────────────────────
def normalize_skills(raw: str) -> list:
    tokens = [t.strip().lower() for t in raw.split(',')]
    canonical = []
    for token in tokens:
        for alias in SORTED_ALIASES:
            if token == alias:
                canonical.append(SKILL_ALIASES[alias])
                break
    seen, result = set(), []
    for s in canonical:
        if s not in seen:
            seen.add(s)
            result.append(s)
    return result


# ─────────────────────────────────────────────
# RESUME DATASET — 10 Candidates
# ─────────────────────────────────────────────
RESUMES_RAW = [
    ("Arjun Sharma",    "Pyhton, MachineLearning, SQL, pandas, numpy, Deep-learning"),
    ("Priya Nair",      "JavaScrpit, Reacts, Node.JS, MongoDb, REST api, HTML/CSS"),
    ("Rahul Gupta",     "Java, Spring Boot, MySql, Microservices, Docker, kubernates"),
    ("Sneha Patel",     "Python, TensorFlow, Keras, NLP, BERT, data-viz, matplotlib"),
    ("Vikram Singh",    "C++, Algoritms, Data Structure, competitive programming, python"),
    ("Ananya Krishnan", "javascript, vue.js, python, flask, PostgreSQL, AWS, CI/CD"),
    ("Karan Mehta",     "Python, Sklearn, XGboost, feature engineering, SQL, tableau"),
    ("Deepika Rao",     "Java, Android, Kotlin, Firebase, REST, UI/UX, figma"),
    ("Aditya Kumar",    "Reactjs, TypeScrpit, GraphQL, redux, tailwind, nodejs, jest"),
    ("Meera Iyer",      "python, R, statistics, ML, regression, clustering, Power-BI"),
]

# ─────────────────────────────────────────────
# JOB DESCRIPTIONS (Required + Preferred combined)
# ─────────────────────────────────────────────
JDS_RAW = {
    "JD-1 — Kakao (ML Engineer)":
        "Python, Machine Learning, Deep Learning, TensorFlow, PyTorch, SQL, "
        "Data Visualization, NLP, BERT, Feature Engineering, Statistics",
    "JD-2 — Naver (Backend Engineer)":
        "Java, Spring Boot, MySQL, PostgreSQL, Microservices, Docker, "
        "Kubernetes, REST API, CI/CD, Redis",
    "JD-3 — Line (Frontend Engineer)":
        "JavaScript, React, Vue, TypeScript, REST API, HTML/CSS, "
        "Node.js, GraphQL, Redux, Jest, AWS",
}


# ─────────────────────────────────────────────
# STEP 3 — Build Vocabulary
# ─────────────────────────────────────────────
normalized = [(name, normalize_skills(raw)) for name, raw in RESUMES_RAW]
all_skills = set(s for _, skills in normalized for s in skills)
VOCAB = sorted(all_skills)
VOCAB_IDX = {skill: i for i, skill in enumerate(VOCAB)}
V = len(VOCAB)

# ─────────────────────────────────────────────
# STEP 4 — TF-IDF Vectors for Resumes
# ─────────────────────────────────────────────
df = {skill: sum(1 for _, skills in normalized if skill in skills) for skill in VOCAB}
idf = {skill: math.log(10 / df[skill]) for skill in VOCAB}

def tfidf_vector(skills: list) -> list:
    N = len(skills)
    vec = [0.0] * V
    for skill in skills:
        vec[VOCAB_IDX[skill]] = (1.0 / N) * idf[skill]
    return vec

resume_vectors = [(name, tfidf_vector(skills)) for name, skills in normalized]

# ─────────────────────────────────────────────
# STEP 5 — JD Binary Vectors
# ─────────────────────────────────────────────
def jd_binary_vector(raw: str) -> list:
    tokens = [t.strip().lower() for t in raw.split(',')]
    vec = [0.0] * V
    for token in tokens:
        for alias in SORTED_ALIASES:
            if token == alias:
                canonical = SKILL_ALIASES[alias]
                if canonical in VOCAB_IDX:
                    vec[VOCAB_IDX[canonical]] = 1.0
                break
    return vec

jd_vectors = {name: jd_binary_vector(raw) for name, raw in JDS_RAW.items()}

# ─────────────────────────────────────────────
# STEP 6 — Cosine Similarity & Ranking
# ─────────────────────────────────────────────
def cosine(a: list, b: list) -> float:
    dot   = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b) if (norm_a and norm_b) else 0.0

print("=" * 55)
print("  RESUME MATCHING ENGINE — FINAL RESULTS")
print("=" * 55)
for jd_name, jd_vec in jd_vectors.items():
    scores = sorted(
        [(name, cosine(rv, jd_vec)) for name, rv in resume_vectors],
        key=lambda x: (-x[1], x[0])
    )
    print(f"\n{jd_name}")
    print(", ".join(f"{n}({s:.2f})" for n, s in scores[:3]))
