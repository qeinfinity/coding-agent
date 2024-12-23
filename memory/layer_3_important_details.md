# Layer 3: Important Implementation Details

## Python Specific Patterns

### Flask API with JWT
```python
from flask import Flask, jsonify, request
from flask_jwt_extended import JWTManager, create_access_token, jwt_required

app = Flask(__name__)
app.config['JWT_SECRET_KEY'] = 'your-secret-key'
jwt = JWTManager(app)

@app.route('/login', methods=['POST'])
def login():
    username = request.json.get('username')
    password = request.json.get('password')
    # Add authentication logic here
    access_token = create_access_token(identity=username)
    return jsonify(access_token=access_token)

@app.route('/protected', methods=['GET'])
@jwt_required()
def protected():
    return jsonify(message='Access granted')
```

### SQLAlchemy Models
```python
from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    username = Column(String(80), unique=True, nullable=False)
    email = Column(String(120), unique=True, nullable=False)
    posts = relationship('Post', backref='author', lazy=True)

class Post(Base):
    __tablename__ = 'posts'
    id = Column(Integer, primary_key=True)
    title = Column(String(100), nullable=False)
    content = Column(String(500), nullable=False)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
```

### FastAPI Async Example
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    name: str
    price: float

@app.post("/items/")
async def create_item(item: Item):
    try:
        # Add database logic here
        return {"message": "Item created", "item": item}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
```

### Pytest Fixtures
```python
import pytest
from your_app import create_app, db

@pytest.fixture
def app():
    app = create_app('testing')
    return app

@pytest.fixture
def client(app):
    return app.test_client()

@pytest.fixture
def db_session(app):
    with app.app_context():
        db.create_all()
        yield db
        db.session.remove()
        db.drop_all()
```

## Common API Response Formats

### REST Success Response
```json
{
    "status": "success",
    "data": {
        "id": 1,
        "name": "Example",
        "created_at": "2023-01-01T00:00:00Z"
    }
}
```

### REST Error Response
```json
{
    "status": "error",
    "error": {
        "code": "VALIDATION_ERROR",
        "message": "Invalid input",
        "details": ["Field 'name' is required"]
    }
}
