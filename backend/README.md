# EmoSense Backend

Flask API with JWT authentication, bcrypt hashing, and MySQL.

## Quick Start

### 1. Database Setup

Open MySQL and run:

```sql
mysql -u root -p < setup.sql
```

Or paste `setup.sql` contents into MySQL Workbench / phpMyAdmin.

### 2. Configure Environment

Edit `backend/.env`:

```
MYSQL_HOST=localhost
MYSQL_USER=root
MYSQL_PASSWORD=YOUR_ACTUAL_PASSWORD
MYSQL_DATABASE=emosense
SECRET_KEY=any-long-random-string-here
```

### 3. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 4. Run the Server

```bash
python app.py
```

Server starts at `http://localhost:5000`.

### 5. Open Frontend

Open `index.html` in your browser. Register an account, then log in.

---

## API Reference

### POST `/register`

```json
// Request
{ "firstName": "Ada", "lastName": "Lovelace", "email": "ada@emosense.io", "password": "securepass123" }

// Response 201
{ "message": "Account created successfully." }

// Response 409
{ "error": "An account with this email already exists." }
```

### POST `/login`

```json
// Request
{ "email": "ada@emosense.io", "password": "securepass123" }

// Response 200
{
  "message": "Login successful.",
  "token": "eyJhbGciOiJIUzI1...",
  "user": { "id": 1, "firstName": "Ada", "lastName": "Lovelace", "email": "ada@emosense.io" }
}

// Response 401
{ "error": "Invalid email or password." }
```

### GET `/profile` (protected)

```
Authorization: Bearer <token>
```

```json
// Response 200
{ "id": 1, "firstName": "Ada", "lastName": "Lovelace", "email": "ada@emosense.io", "createdAt": "2026-03-20T20:30:00" }
```
