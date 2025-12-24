"""
Meta Ads Compliance Scanner - Complete Backend
FastAPI + PostgreSQL + Claude AI

SETUP INSTRUCTIONS:
1. Install dependencies: pip install fastapi uvicorn sqlalchemy psycopg2-binary anthropic python-multipart pillow python-dotenv pydantic-settings
2. Create .env file with:
   - DATABASE_URL=postgresql://user:password@localhost/metaads
   - ANTHROPIC_API_KEY=your_key_here
   - SECRET_KEY=your_secret_key_here
3. Run: uvicorn main:app --reload
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime
import anthropic
import base64
import os
from typing import Optional, List
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./metaads.db")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# Models
class ScanResult(Base):
    __tablename__ = "scan_results"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    ad_copy = Column(Text)
    image_name = Column(String, nullable=True)
    compliance_score = Column(Float)
    risk_level = Column(String)
    violations = Column(Text)
    recommendations = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# Pydantic models
class ScanResponse(BaseModel):
    scan_id: int
    compliance_score: float
    risk_level: str
    violations: List[dict]
    recommendations: List[str]
    summary: str

# FastAPI app
app = FastAPI(title="Meta Ads Compliance Scanner API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Anthropic client
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Meta Policy Rules (comprehensive database)
META_POLICIES = """
META ADVERTISING POLICIES - COMPREHENSIVE RULES:

PROHIBITED CONTENT:
1. Illegal Products or Services
2. Discriminatory Practices (targeting based on personal attributes)
3. Tobacco and Related Products
4. Drugs & Drug-Related Products
5. Unsafe Supplements (anabolic steroids, chitosan, ephedra, HCG)
6. Weapons, Ammunition, Explosives
7. Adult Products and Services
8. Adult Content (nudity, sexual content)
9. Third-Party Infringement (copyright, trademark violations)
10. Sensational Content (gore, violence, shocking imagery)
11. Personal Attributes (cannot assert or imply personal attributes)
12. Misinformation and False News
13. Controversial Content (exploiting crises, political issues)

RESTRICTED CONTENT (Requires Special Authorization):
1. Alcohol - Must comply with local laws, age restrictions
2. Dating Services - Cannot be sexually suggestive
3. Real Money Gambling - Requires written permission
4. State Lotteries - Requires written permission
5. Online Pharmacies - Requires LegitScript certification
6. Supplements - Cannot make drug-like claims
7. Subscription Services - Must clearly disclose terms
8. Financial Services - Must show legal disclosures
9. Credit Services - Comply with regulations
10. Employment Opportunities - Cannot contain misleading info
11. Housing - Cannot discriminate

BEFORE/AFTER IMAGES:
- Before and after images are generally NOT allowed
- Cannot show unexpected or unlikely results
- Cannot imply time-based results without substantiation

HEALTH & WEIGHT LOSS:
- Cannot claim to cure, treat, or prevent diseases
- Cannot use "magic pill" or "miracle cure" language
- Cannot show unrealistic body transformations
- Cannot target based on health conditions
- Avoid words: cure, treat, diagnose, prevent, FDA-approved (unless true)

FINANCIAL CLAIMS:
- Cannot guarantee financial results
- Must include risk disclosures for investments
- Cannot promise "get rich quick"
- Avoid: "guaranteed returns", "risk-free", "easy money"

PERSONAL ATTRIBUTES:
- Cannot say "Are you overweight?" or "Losing your hair?"
- Cannot directly address personal characteristics
- Cannot imply you know something about the viewer

SENSATIONAL/CLICKBAIT:
- Cannot use shocking or sensational imagery
- Cannot use exaggerated claims
- Cannot withhold information to force clicks
- Avoid ALL CAPS in headlines excessively

IMAGE REQUIREMENTS:
- Text in images should be minimal (avoid more than 20% text)
- No graphic violence or blood
- No sexually suggestive content
- No shocking or disgusting imagery
- High quality, not pixelated
- Cannot show non-functional landing pages

PROHIBITED WORDS/PHRASES:
- "FREE" without clear terms
- "Limited time" without actual limitation
- Medical claims without proof
- Income guarantees
- "As seen on TV" without proof
- "FDA approved" (for supplements)
- "Cure", "Treat", "Heal" for health products
- Comparison to competitors without substantiation

TARGETING RESTRICTIONS:
- Cannot target based on sensitive categories
- Health conditions
- Financial status
- Sexual orientation (in some regions)
- Political affiliation (restricted)

LANDING PAGE REQUIREMENTS:
- Must match ad content
- Must be functional
- Must have privacy policy
- Must have clear pricing
- Must not use pop-ups that interfere with navigation
"""

def analyze_with_ai(ad_copy: str, image_data: Optional[str] = None) -> dict:
    """Use Claude to analyze ad compliance"""
    
    messages_content = []
    
    # Add image if provided
    if image_data:
        messages_content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": image_data
            }
        })
    
    # Add text analysis prompt
    prompt = f"""You are a Meta Ads compliance expert. Analyze this advertisement for policy violations.

AD COPY:
{ad_copy}

META POLICIES TO CHECK AGAINST:
{META_POLICIES}

Provide a detailed analysis in the following JSON format:
{{
    "compliance_score": <0-100, where 100 is fully compliant>,
    "risk_level": "<LOW|MEDIUM|HIGH|CRITICAL>",
    "violations": [
        {{
            "category": "<policy category>",
            "severity": "<LOW|MEDIUM|HIGH>",
            "issue": "<specific problem>",
            "text_snippet": "<the problematic text>",
            "policy_reference": "<which Meta policy this violates>"
        }}
    ],
    "recommendations": [
        "<specific actionable fix>"
    ],
    "summary": "<brief overall assessment>"
}}

Be thorough and specific. Flag even potential issues. If the ad is compliant, say so clearly."""

    messages_content.append({
        "type": "text",
        "text": prompt
    })
    
    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[{
                "role": "user",
                "content": messages_content
            }]
        )
        
        # Extract JSON from response
        response_text = response.content[0].text
        
        # Find JSON in the response
        import json
        import re
        
        # Try to extract JSON from markdown code blocks or raw text
        json_match = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find JSON object directly
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = response_text
        
        result = json.loads(json_str)
        return result
        
    except Exception as e:
        # Fallback response if AI fails
        return {
            "compliance_score": 50,
            "risk_level": "MEDIUM",
            "violations": [{
                "category": "Analysis Error",
                "severity": "MEDIUM",
                "issue": f"Could not complete analysis: {str(e)}",
                "text_snippet": "",
                "policy_reference": "System Error"
            }],
            "recommendations": ["Please try again or contact support"],
            "summary": "Analysis could not be completed due to technical error"
        }

@app.get("/")
def root():
    return {
        "message": "Meta Ads Compliance Scanner API",
        "version": "1.0",
        "endpoints": {
            "/scan": "POST - Scan ad copy and image",
            "/history/{user_id}": "GET - Get scan history",
            "/health": "GET - Health check"
        }
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@app.post("/scan", response_model=ScanResponse)
async def scan_ad(
    ad_copy: str = Form(...),
    user_id: str = Form(default="anonymous"),
    image: Optional[UploadFile] = File(None)
):
    """
    Scan an advertisement for Meta compliance issues
    
    Parameters:
    - ad_copy: The text content of the ad
    - user_id: User identifier (optional)
    - image: Ad image file (optional)
    """
    
    # Process image if provided
    image_data = None
    image_name = None
    
    if image:
        image_name = image.filename
        image_bytes = await image.read()
        image_data = base64.b64encode(image_bytes).decode('utf-8')
    
    # Analyze with AI
    analysis = analyze_with_ai(ad_copy, image_data)
    
    # Save to database
    db = SessionLocal()
    try:
        scan_result = ScanResult(
            user_id=user_id,
            ad_copy=ad_copy,
            image_name=image_name,
            compliance_score=analysis["compliance_score"],
            risk_level=analysis["risk_level"],
            violations=str(analysis["violations"]),
            recommendations=str(analysis["recommendations"]),
        )
        db.add(scan_result)
        db.commit()
        db.refresh(scan_result)
        
        return ScanResponse(
            scan_id=scan_result.id,
            compliance_score=analysis["compliance_score"],
            risk_level=analysis["risk_level"],
            violations=analysis["violations"],
            recommendations=analysis["recommendations"],
            summary=analysis["summary"]
        )
    finally:
        db.close()

@app.get("/history/{user_id}")
def get_scan_history(user_id: str, limit: int = 10):
    """Get scan history for a user"""
    db = SessionLocal()
    try:
        scans = db.query(ScanResult)\
            .filter(ScanResult.user_id == user_id)\
            .order_by(ScanResult.created_at.desc())\
            .limit(limit)\
            .all()
        
        return {
            "user_id": user_id,
            "total_scans": len(scans),
            "scans": [{
                "scan_id": s.id,
                "compliance_score": s.compliance_score,
                "risk_level": s.risk_level,
                "created_at": s.created_at.isoformat(),
                "ad_copy_preview": s.ad_copy[:100] + "..." if len(s.ad_copy) > 100 else s.ad_copy
            } for s in scans]
        }
    finally:
        db.close()

@app.get("/scan/{scan_id}")
def get_scan_details(scan_id: int):
    """Get detailed results for a specific scan"""
    db = SessionLocal()
    try:
        scan = db.query(ScanResult).filter(ScanResult.id == scan_id).first()
        
        if not scan:
            raise HTTPException(status_code=404, detail="Scan not found")
        
        import ast
        
        return {
            "scan_id": scan.id,
            "ad_copy": scan.ad_copy,
            "compliance_score": scan.compliance_score,
            "risk_level": scan.risk_level,
            "violations": ast.literal_eval(scan.violations),
            "recommendations": ast.literal_eval(scan.recommendations),
            "created_at": scan.created_at.isoformat()
        }
    finally:
        db.close()

@app.get("/policies")
def get_policies():
    """Get Meta advertising policies reference"""
    return {
        "policies": META_POLICIES,
        "categories": [
            "Prohibited Content",
            "Restricted Content",
            "Before/After Images",
            "Health & Weight Loss",
            "Financial Claims",
            "Personal Attributes",
            "Image Requirements"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
