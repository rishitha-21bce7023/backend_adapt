from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

app = Flask(__name__)
CORS(app)

client = MongoClient("mongodb+srv://kodavati009:harsha@cluster1.ak54w.mongodb.net/modules?retryWrites=true&w=majority")
db = client['modules']
questions_collection = db['module_questions']

class AdaptiveELearningModel:
    def __init__(self):
        self.user_classifier = Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestClassifier(n_estimators=200, random_state=42))
        ])
        
        self.module_skip_classifier = Pipeline([
            ('scaler', StandardScaler()),
            ('mlp', MLPClassifier(hidden_layer_sizes=(100, 50, 25), max_iter=2000, random_state=42))
        ])
    
    def train_models(self):
        np.random.seed(42)
        n_samples = 1000
        
        X_users = np.column_stack([
            np.random.randint(15, 60, n_samples),
            np.random.randint(0, 101, n_samples)
        ])
        y_users = np.where(X_users[:, 1] < 50, 'basic', 
                           np.where(X_users[:, 1] < 75, 'intermediate', 'advanced'))

        X_modules = np.column_stack([
            np.random.randint(1, 4, n_samples),
            np.random.randint(0, 101, n_samples)
        ])
        y_modules = (X_modules[:, 1] >= 75) | ((X_modules[:, 0] == 3) & (X_modules[:, 1] >= 60))
        
        X_users_train, X_users_test, y_users_train, y_users_test = train_test_split(X_users, y_users, test_size=0.2, random_state=42)
        X_modules_train, X_modules_test, y_modules_train, y_modules_test = train_test_split(X_modules, y_modules, test_size=0.2, random_state=42)
        
        self.user_classifier.fit(X_users_train, y_users_train)
        user_pred = self.user_classifier.predict(X_users_test)
        
        self.module_skip_classifier.fit(X_modules_train, y_modules_train)
        module_pred = self.module_skip_classifier.predict(X_modules_test)
    
    def classify_user(self, user_features):
        return self.user_classifier.predict([user_features])[0]
    
    def can_skip_module(self, module_features):
        return bool(self.module_skip_classifier.predict([module_features])[0])

model = AdaptiveELearningModel()

@app.route('/categories', methods=['GET'])
def get_categories():
    categories = list(questions_collection.distinct("category"))
    return jsonify([{"id": cat, "name": cat} for cat in categories])

@app.route('/modules/<category>', methods=['GET'])
def get_modules(category):
    modules = list(questions_collection.find(
        {"category": category},
        {"module_id": 1, "module_name": 1, "_id": 0}
    ))
    return jsonify(modules)

@app.route('/questions/<module_id>', methods=['GET'])
def get_questions(module_id):
    module = questions_collection.find_one({"module_id": module_id}, {"questions": 1, "_id": 0})
    if module:
        return jsonify(module['questions'])
    return jsonify([])

@app.route('/submit_quiz', methods=['POST'])
def submit_quiz():
    data = request.json
    user_answers = data['user_answers']
    module_id = data['module_id']

    module = questions_collection.find_one({"module_id": module_id}, {"questions": 1, "category": 1, "_id": 0})
    questions = module['questions']
    category = module['category']

    score = sum(1 for user_ans in user_answers 
                for q in questions 
                if q['question_id'] == user_ans['question_id'] and q['correct_option'] == user_ans['answer'])
    
    total_questions = len(questions)
    percentage_score = (score / total_questions) * 100

    time_taken = data.get('time_taken', 30)
    user_features = [time_taken, percentage_score]
    classification = model.classify_user(user_features)

    user_level = {'basic': 1, 'intermediate': 2, 'advanced': 3}[classification]
    module_features = [user_level, percentage_score]

    can_skip = percentage_score == 100 or model.can_skip_module(module_features)

    all_modules = list(questions_collection.find(
        {"category": category},
        {"module_id": 1, "module_name": 1, "_id": 0}
    ))
    current_module_index = next((i for i, m in enumerate(all_modules) if m['module_id'] == module_id), -1)
    next_module = all_modules[current_module_index + 1] if current_module_index < len(all_modules) - 1 else None

    return jsonify({
        'classification': classification,
        'can_skip': can_skip,
        'score': percentage_score,
        'explanation': f"Your score is {percentage_score}% and you have been classified as {classification} level. You {'can' if can_skip else 'cannot'} skip the next module.",
        'next_module': next_module
    })

@app.route('/lessons/<module_id>', methods=['GET'])
def get_lessons(module_id):
    module = questions_collection.find_one({"module_id": module_id}, {"lessons": 1, "_id": 0})
    lessons = module.get("lessons", []) if module else []
    return jsonify(lessons)

if __name__ == '__main__':
    model.train_models()
    app.run(debug=True)
