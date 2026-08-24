from flask import Flask, render_template, request, redirect, url_for, session, flash
import sqlite3


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)

app.secret_key = "ai_traffic_parking_secret_key"


# ============================================================
# DATABASE
# ============================================================

DATABASE = "users.db"


def get_db():

    conn = sqlite3.connect(DATABASE)

    conn.row_factory = sqlite3.Row

    return conn


# ============================================================
# CREATE USERS TABLE
# ============================================================

def create_database():

    conn = get_db()

    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            fullname TEXT NOT NULL,

            email TEXT UNIQUE NOT NULL,

            mobile TEXT NOT NULL,

            location TEXT NOT NULL,

            age INTEGER NOT NULL,

            password TEXT NOT NULL

        )
    """)

    conn.commit()

    conn.close()


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
@app.route("/home")
def home():

    return render_template("home.html")


# ============================================================
# SIGN UP
# ============================================================

@app.route("/signup", methods=["GET", "POST"])
def signup():

    if request.method == "POST":

        fullname = request.form.get("fullname")
        email = request.form.get("email")
        mobile = request.form.get("mobile")
        location = request.form.get("location")
        age = request.form.get("age")
        password = request.form.get("password")
        confirm_password = request.form.get("confirm_password")


        # Remove extra spaces

        fullname = fullname.strip()
        email = email.strip().lower()
        mobile = mobile.strip()
        location = location.strip()


        # ------------------------------------------------
        # AGE CHECK
        # ------------------------------------------------

        try:

            age = int(age)

        except:

            flash("Please enter a valid age.")

            return redirect(url_for("signup"))


        if age < 21:

            flash("You must be at least 21 years old.")

            return redirect(url_for("signup"))


        # ------------------------------------------------
        # PASSWORD CHECK
        # ------------------------------------------------

        if password != confirm_password:

            flash("Passwords do not match.")

            return redirect(url_for("signup"))


        if len(password) < 6:

            flash("Password must contain at least 6 characters.")

            return redirect(url_for("signup"))


        # ------------------------------------------------
        # MOBILE CHECK
        # ------------------------------------------------

        if not mobile.isdigit() or len(mobile) != 10:

            flash("Please enter a valid 10-digit mobile number.")

            return redirect(url_for("signup"))


        # ------------------------------------------------
        # DATABASE
        # ------------------------------------------------

        conn = get_db()

        cursor = conn.cursor()


        # Check email

        cursor.execute(
            "SELECT * FROM users WHERE email = ?",
            (email,)
        )

        existing_user = cursor.fetchone()


        if existing_user:

            conn.close()

            flash("Email already registered. Please login.")

            return redirect(url_for("login"))


        # ------------------------------------------------
        # INSERT USER
        # ------------------------------------------------

        cursor.execute("""
            INSERT INTO users
            (
                fullname,
                email,
                mobile,
                location,
                age,
                password
            )

            VALUES (?, ?, ?, ?, ?, ?)

        """, (
            fullname,
            email,
            mobile,
            location,
            age,
            password
        ))


        conn.commit()

        conn.close()


        # ------------------------------------------------
        # SIGNUP SUCCESS
        # ------------------------------------------------

        flash("Account created successfully. Please login.")

        return redirect(url_for("login"))


    return render_template("signup.html")


# ============================================================
# LOGIN
# ============================================================

@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        conn = get_db()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM users WHERE email = ?",
            (email,)
        )

        user = cursor.fetchone()

        conn.close()

        # Email not found
        if user is None:
            flash("Email not registered.")
            return redirect(url_for("login"))

        # Password incorrect
        if user["password"] != password:
            flash("Incorrect password.")
            return redirect(url_for("login"))

        # ==========================================
        # LOGIN SUCCESS
        # ==========================================

        session.clear()

        session["logged_in"] = True
        session["user_id"] = user["id"]
        session["fullname"] = user["fullname"]
        session["email"] = user["email"]
        session["location"] = user["location"]

        # Go directly to prediction page
        return redirect(url_for("predict"))

    return render_template("login.html")


# ============================================================
# VERIFY PAGE
# ============================================================

@app.route("/verify")
def verify():

    # User must login first

    if not session.get("logged_in"):

        return redirect(url_for("login"))


    return render_template("verify.html")


# ============================================================
# PREDICT PAGE
# ============================================================

@app.route("/predict", methods=["GET", "POST"])
def predict():

    if not session.get("logged_in"):
        return redirect(url_for("login"))

    if request.method == "POST":

        location = request.form.get("location")
        date = request.form.get("date")
        time = request.form.get("time")

        vehicle_count = request.form.get("vehicle_count")
        average_speed = request.form.get("average_speed")

        algorithm = request.form.get("algorithm")

        # Temporary prediction
        prediction = "Medium"

        predicted_occupancy = "65%"

        congestion_probability = "72%"

        return render_template(
            "predict.html",
            prediction=prediction,
            predicted_occupancy=predicted_occupancy,
            congestion_probability=congestion_probability,
            algorithm=algorithm
        )

    return render_template("predict.html")


# ============================================================
# LOGOUT
# ============================================================

@app.route("/logout")
def logout():

    session.clear()

    return redirect(url_for("login"))


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    # Create database

    create_database()


    print("----------------------------------------")

    print("AI Traffic & Parking Intelligence")

    print("----------------------------------------")

    print("Application running at:")

    print("http://127.0.0.1:5000")

    print("----------------------------------------")


    app.run(debug=True)
