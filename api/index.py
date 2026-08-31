import sys
import os

# Add project root folder to Python path
sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from web import app, create_database

# Create database when Vercel starts the function
create_database()
