from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

app = Flask(__name__)
app.secret_key = "supersecretkey"
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

class Rating(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('student.id'), nullable=False)
    course_id = db.Column(db.Integer, db.ForeignKey('course.id'), nullable=False)
    rating = db.Column(db.Integer, nullable=False)  # Rating (1-5)
    comment = db.Column(db.Text, nullable=False)  # Student's comment

    student = db.relationship('Student', backref=db.backref('ratings', lazy=True))
    course = db.relationship('Course', backref=db.backref('ratings', lazy=True))
   

class Course(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text, nullable=False)
    duration = db.Column(db.String(100), nullable=False)
    skills_covered = db.Column(db.String(500), nullable=False)  # New field for skills
    institute_id = db.Column(db.Integer, db.ForeignKey('institute.id'), nullable=False)
    institute = db.relationship('Institute', backref=db.backref('courses', lazy=True))
    blocked = db.Column(db.Boolean, default=False)

class Enrollment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('student.id'), nullable=False)
    course_id = db.Column(db.Integer, db.ForeignKey('course.id'), nullable=False)
    institute_id = db.Column(db.Integer, db.ForeignKey('institute.id'), nullable=False)

    student = db.relationship('Student', backref=db.backref('enrollments', lazy=True))
    course = db.relationship('Course', backref=db.backref('enrolled_students', lazy=True))
    institute = db.relationship('Institute', backref=db.backref('enrollments', lazy=True))

# Student Model
class Student(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    skills = db.Column(db.String(500), nullable=False)
    interest_area = db.Column(db.String(200), nullable=False)
    blocked = db.Column(db.Boolean, default=False)

# Institute Model
class Institute(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    blocked = db.Column(db.Boolean, default=False)

# Home Page
@app.route('/')
def index():
    return render_template('index.html')

# Student Registration
@app.route('/register/student', methods=['GET', 'POST'])
def register_student():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = generate_password_hash(request.form['password'])
        skills = ','.join(request.form.getlist('skills'))
        interest_area = request.form['interest_area']

        student = Student(name=name, email=email, password=password, skills=skills, interest_area=interest_area)
        db.session.add(student)
        db.session.commit()
        flash('Student Registered Successfully!', 'success')
        return redirect(url_for('login'))

    return render_template('register_student.html')

# Institute Registration
@app.route('/register/institute', methods=['GET', 'POST'])
def register_institute():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = generate_password_hash(request.form['password'])

        institute = Institute(name=name, email=email, password=password)
        db.session.add(institute)
        db.session.commit()
        flash('Institute Registered Successfully!', 'success')
        return redirect(url_for('login'))

    return render_template('register_institute.html')

# Login
# Hardcoded Admin Credentials
ADMIN_EMAIL = "admin@admin.com"
ADMIN_PASSWORD = "admin123"

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user_type = request.form['user_type']

        # Admin Login Check
        if user_type == 'admin':
            if email == ADMIN_EMAIL and password == ADMIN_PASSWORD:
                session['user'] = email
                session['user_type'] = 'admin'
                flash("Welcome, Admin!", 'success')
                return redirect(url_for('dashboard_admin'))
            else:
                flash("Invalid admin credentials!", 'danger')
                return redirect(url_for('login'))

        # Student or Institute Login
        if user_type == 'student':
            user = Student.query.filter_by(email=email).first()
            dashboard = 'dashboard_student'
        elif user_type == 'institute':
            user = Institute.query.filter_by(email=email).first()
            dashboard = 'dashboard_institute'
        else:
            user = None
            dashboard = None

        if user and check_password_hash(user.password, password):
            if hasattr(user, 'blocked') and user.blocked:
                flash("Your account has been blocked by admin!", 'danger')
                return redirect(url_for('login'))

            session['user'] = user.email
            session['user_type'] = user_type
            flash(f"Welcome, {user.name}!", 'success')
            return redirect(url_for(dashboard))
        else:
            flash("Invalid credentials, please try again!", 'danger')

    return render_template('login.html')

def recommend_courses_graph(student_id):
    G = build_knowledge_graph()
    
    if f"student_{student_id}" not in G:
        return []

    # Compute Personalized PageRank
    pagerank_scores = nx.pagerank(G, personalization={f"student_{student_id}": 1})

    # Rank courses based on scores
    course_scores = {node: score for node, score in pagerank_scores.items() if "course_" in node}
    sorted_courses = sorted(course_scores.items(), key=lambda x: x[1], reverse=True)

    # Debugging: Print scaled course scores
    print("\nScaled Course Scores (Multiplied by 10) ")
    for course_node, score in course_scores.items():
        print(f"{course_node}: {score * 1000:.2f}%")

    # Fetch course details
    recommended_courses = []
    for course_node, score in sorted_courses[:10]:  
        course_id = int(course_node.split("_")[1])
        course = db.session.get(Course, course_id)

        clean_score = float(score) * 1000 if score is not None else 0.0

        recommended_courses.append({"course": course, "score": clean_score})

    return recommended_courses


@app.route('/dashboard/admin')
def dashboard_admin():
    if 'user' not in session or session['user_type'] != 'admin':
        flash('Unauthorized access!', 'danger')
        return redirect(url_for('login'))

    students = Student.query.all()
    institutes = Institute.query.all()
    courses = Course.query.all()
    return render_template('dashboard_admin.html', students=students, institutes=institutes, courses=courses)

@app.route('/block/student/<int:student_id>')
def block_student(student_id):
    student = Student.query.get_or_404(student_id)
    student.blocked = not student.blocked
    db.session.commit()
    return redirect(url_for('dashboard_admin'))

@app.route('/block/institute/<int:institute_id>')
def block_institute(institute_id):
    institute = Institute.query.get_or_404(institute_id)
    institute.blocked = not institute.blocked
    db.session.commit()
    return redirect(url_for('dashboard_admin'))

@app.route('/block/course/<int:course_id>')
def block_course(course_id):
    course = Course.query.get_or_404(course_id)
    course.blocked = not course.blocked
    db.session.commit()
    return redirect(url_for('dashboard_admin'))



@app.route('/dashboard/student')
def dashboard_student():
    if 'user' not in session or session['user_type'] != 'student':
        flash('Unauthorized access!', 'danger')
        return redirect(url_for('login'))

    student = Student.query.filter_by(email=session['user']).first()

    if student.blocked:
        flash("Your account has been blocked!", "danger")
        return redirect(url_for('logout'))
    
    courses = Course.query.all()

    enrolled_courses = student.enrollments
    enrolled_course_ids = {enrollment.course_id for enrollment in enrolled_courses}

    # Graph-based recommendation system
    recommended_courses = recommend_courses_graph(student.id)

    #  Debugging: Print recommended courses and scores
    print("\nRecommended Courses Debugging ")
    for item in recommended_courses:
        print(f"Course: {item['course'].name}, Score: {item['score']}")

    # Separate recommended and other courses
    recommended_courses_filtered = []
    other_courses = []

    for item in recommended_courses:
        if item["score"] > 0.01:  # Only recommend courses with meaningful relevance
            recommended_courses_filtered.append(item)
        else:
            other_courses.append(item)

    # Calculate ratings and comments for each course
    course_ratings = {}
    course_comments = {}

    for course in courses:
        ratings = [rating.rating for rating in course.ratings]
        course_ratings[course.id] = round(sum(ratings) / len(ratings), 1) if ratings else "No ratings yet"
        course_comments[course.id] = course.ratings

    return render_template(
        'dashboard_student.html',
        recommended_courses=recommended_courses_filtered,
        other_courses=other_courses,
        enrolled_courses=enrolled_courses,
        enrolled_course_ids=enrolled_course_ids,
        course_ratings=course_ratings,
        course_comments=course_comments
    )

@app.route('/dashboard/institute', methods=['GET', 'POST'])
def dashboard_institute():
    if 'user' not in session or session['user_type'] != 'institute':
        flash('Unauthorized access!', 'danger')
        return redirect(url_for('login'))

    institute = Institute.query.filter_by(email=session['user']).first()

    # Handle adding a course (POST method)
    if request.method == 'POST':
        course_name = request.form['course_name']
        description = request.form['description']
        duration = request.form['duration']
        skills_covered = request.form['skills_covered']

        new_course = Course(name=course_name, description=description, duration=duration, 
                            skills_covered=skills_covered, institute_id=institute.id)
        db.session.add(new_course)
        db.session.commit()
        flash("Course added successfully!", "success")
        return redirect(url_for('dashboard_institute'))  # Refresh page after adding course

    # Fetch institute courses & ratings
    courses = Course.query.filter_by(institute_id=institute.id).all()
    
    course_ratings = {}
    course_comments = {}

    for course in courses:
        ratings = [r.rating for r in course.ratings]  # Get list of ratings

        # Fix: Ensure only numbers are passed to `round()`
        if ratings:  
            avg_rating = sum(ratings) / len(ratings)  # Compute average rating
            course_ratings[course.id] = round(avg_rating, 1)  # Round it correctly
        else:
            course_ratings[course.id] = "No ratings yet"  # Assign a string if no ratings

        course_comments[course.id] = course.ratings  # Store comments

    return render_template('dashboard_institute.html', 
                           courses=courses, 
                           course_ratings=course_ratings, 
                           course_comments=course_comments)

@app.route('/my-enrolled-courses')
def my_enrolled_courses():
    if 'user' not in session or session['user_type'] != 'student':
        flash('Unauthorized access!', 'danger')
        return redirect(url_for('login'))

    student = Student.query.filter_by(email=session['user']).first()
    enrolled_courses = student.enrollments  # Get all enrolled courses

    # Prepare data for ratings and comments
    course_ratings = {}
    course_comments = {}

    for enrollment in enrolled_courses:
        course = enrollment.course
        ratings = [r.rating for r in course.ratings]
        course_ratings[course.id] = round(sum(ratings) / len(ratings), 1) if ratings else "No ratings yet"
        course_comments[course.id] = course.ratings

    return render_template('enrolled_courses.html', 
                           enrolled_courses=enrolled_courses, 
                           course_ratings=course_ratings, 
                           course_comments=course_comments)


@app.route('/enroll/<int:course_id>', methods=['GET', 'POST'])
def enroll(course_id):
    if 'user' not in session or session['user_type'] != 'student':
        flash('Unauthorized access!', 'danger')
        return redirect(url_for('login'))

    student = Student.query.filter_by(email=session['user']).first()
    if student.blocked:
        flash("Your account has been blocked!", "danger")
        return redirect(url_for('logout'))

    course = Course.query.get(course_id)

    #  Check if the course is blocked
    if course.blocked:
        flash("This course has been blocked by the admin and cannot be enrolled!", "danger")
        return redirect(url_for('dashboard_student'))

    institute = course.institute  

    # Check if the student is already enrolled
    existing_enrollment = Enrollment.query.filter_by(student_id=student.id, course_id=course.id).first()
    if existing_enrollment:
        flash(f"You are already enrolled in {course.name}!", "warning")
        return redirect(url_for('dashboard_student'))

    if request.method == 'POST':  # Dummy Payment
        new_enrollment = Enrollment(student_id=student.id, course_id=course.id, institute_id=institute.id)
        db.session.add(new_enrollment)
        db.session.commit()
        flash(f"You have successfully enrolled in {course.name} from {institute.name}!", "success")
        return redirect(url_for('my_enrolled_courses'))  

    # Fetch course ratings and comments
    ratings = [rating.rating for rating in course.ratings]
    course_rating = round(sum(ratings) / len(ratings), 1) if ratings else "No ratings yet"
    course_comments = course.ratings  

    return render_template('enroll.html', course=course, institute=institute, course_rating=course_rating, course_comments=course_comments)

@app.route('/rate/<int:course_id>', methods=['GET', 'POST'])
def rate_course(course_id):
    if 'user' not in session or session['user_type'] != 'student':
        flash('Unauthorized access!', 'danger')
        return redirect(url_for('login'))

    student = Student.query.filter_by(email=session['user']).first()
    if student.blocked:
        flash("Your account has been blocked!", "danger")
        return redirect(url_for('logout'))

    course = Course.query.get(course_id)

    # Check if the student is enrolled in this course
    enrollment = Enrollment.query.filter_by(student_id=student.id, course_id=course.id).first()
    if not enrollment:
        flash("You are not enrolled in this course!", "danger")
        return redirect(url_for('dashboard_student'))

    if request.method == 'POST':
        rating_value = int(request.form['rating'])
        comment = request.form['comment']
        
        # Check if the student has already rated this course
        existing_rating = Rating.query.filter_by(student_id=student.id, course_id=course.id).first()
        if existing_rating:
            existing_rating.rating = rating_value  # Update rating
            existing_rating.comment = comment  # Update comment
        else:
            new_rating = Rating(student_id=student.id, course_id=course.id, rating=rating_value, comment=comment)
            db.session.add(new_rating)

        db.session.commit()
        flash(f"You have rated {course.name} with {rating_value} stars!", "success")
        return redirect(url_for('my_enrolled_courses'))

    return render_template('rate_course.html', course=course)

import networkx as nx

def build_knowledge_graph():
    G = nx.Graph()

    # Add Student Nodes
    students = Student.query.all()
    for student in students:
        G.add_node(f"student_{student.id}", type="student")

    # Add Course Nodes
    courses = Course.query.all()
    for course in courses:
        G.add_node(f"course_{course.id}", type="course")
        skills = course.skills_covered.split(',')
        for skill in skills:
            G.add_node(skill.strip(), type="skill")
            G.add_edge(f"course_{course.id}", skill.strip(), relation="covers")

    # Add Student-Interest and Enrollments
    for student in students:
        interests = student.interest_area.split(',')
        for interest in interests:
            G.add_edge(f"student_{student.id}", interest.strip(), relation="interested_in")

        enrollments = Enrollment.query.filter_by(student_id=student.id).all()
        for enrollment in enrollments:
            G.add_edge(f"student_{student.id}", f"course_{enrollment.course_id}", relation="enrolled_in")

    return G

@app.route('/upload_job', methods=['GET', 'POST'])
def upload_job():
    if 'user' not in session or session['user_type'] != 'admin':
        flash('Unauthorized access!', 'danger')
        return redirect(url_for('login'))

    if request.method == 'POST':
        job_title = request.form['job_title']
        job_description = request.form['job_description']
        location = request.form['location']

        # TODO: Save job to database (optional)

        flash("Job uploaded successfully!", "success")
        return redirect(url_for('dashboard_admin'))

    return render_template('job_upload.html')

# Logout
@app.route('/logout')
def logout():
    session.pop('user', None)
    session.pop('user_type', None)
    flash('Logged out successfully!', 'success')
    return redirect(url_for('login'))

if __name__ == '__main__':
    with app.app_context():  # FIXED: Ensure application context before creating tables
        db.create_all()

    app.run(debug=True)
