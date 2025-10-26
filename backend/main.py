from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from io import BytesIO
import base64
import json
import os
from dotenv import load_dotenv
import logging
import re
import pandas as pd
from openai import OpenAI
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Load OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.error("⚠️ OPENAI_API_KEY environment variable not set")
    raise RuntimeError("OPENAI_API_KEY environment variable not set. Please create a .env file with OPENAI_API_KEY=your-key")

# Initialize OpenAI client
try:
    client = OpenAI(api_key=OPENAI_API_KEY)
    logger.info("✅ OpenAI client initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize OpenAI client: {str(e)}")
    raise RuntimeError(f"Failed to initialize OpenAI client: {str(e)}")

app = FastAPI(title="Insurance Policy Processing System")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://sompo-image-report.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Embedded Formula Data
FORMULA_DATA = [
    {"LOB": "TW", "SEGMENT": "1+5", "INSURER": "All Companies", "PO": "90% of Payin", "REMARKS": "NIL"},
    {"LOB": "TW", "SEGMENT": "TW SAOD + COMP", "INSURER": "All Companies", "PO": "90% of Payin", "REMARKS": "NIL"},
    {"LOB": "TW", "SEGMENT": "TW SAOD + COMP", "INSURER": "DIGIT", "PO": "-2%", "REMARKS": "Payin Below 20%"},
    {"LOB": "TW", "SEGMENT": "TW SAOD + COMP", "INSURER": "DIGIT", "PO": "-3%", "REMARKS": "Payin 21% to 30%"},
    {"LOB": "TW", "SEGMENT": "TW SAOD + COMP", "INSURER": "DIGIT", "PO": "-4%", "REMARKS": "Payin 31% to 50%"},
    {"LOB": "TW", "SEGMENT": "TW SAOD + COMP", "INSURER": "DIGIT", "PO": "-5%", "REMARKS": "Payin Above 50%"},
    {"LOB": "TW", "SEGMENT": "TW TP", "INSURER": "Bajaj, Digit, ICICI", "PO": "-2%", "REMARKS": "Payin Below 20%"},
    {"LOB": "TW", "SEGMENT": "TW TP", "INSURER": "Bajaj, Digit, ICICI", "PO": "-3%", "REMARKS": "Payin Above 20%"},
    {"LOB": "TW", "SEGMENT": "TW TP", "INSURER": "Rest of Companies", "PO": "-2%", "REMARKS": "Payin Below 20%"},
    {"LOB": "TW", "SEGMENT": "TW TP", "INSURER": "Rest of Companies", "PO": "-3%", "REMARKS": "Payin 21% to 30%"},
    {"LOB": "TW", "SEGMENT": "TW TP", "INSURER": "Rest of Companies", "PO": "-4%", "REMARKS": "Payin 31% to 50%"},
    {"LOB": "TW", "SEGMENT": "TW TP", "INSURER": "Rest of Companies", "PO": "-5%", "REMARKS": "Payin Above 50%"},
    {"LOB": "PVT CAR", "SEGMENT": "PVT CAR COMP + SAOD", "INSURER": "All Companies", "PO": "90% of Payin", "REMARKS": "All Fuel"},
    {"LOB": "PVT CAR", "SEGMENT": "PVT CAR TP", "INSURER": "Bajaj, Digit, SBI", "PO": "-2%", "REMARKS": "Payin Below 20%"},
    {"LOB": "PVT CAR", "SEGMENT": "PVT CAR TP", "INSURER": "Bajaj, Digit, SBI", "PO": "-3%", "REMARKS": "Payin Above 20%"},
    {"LOB": "PVT CAR", "SEGMENT": "PVT CAR TP", "INSURER": "Rest of Companies", "PO": "90% of Payin", "REMARKS": "Zuno -  21"},
    {"LOB": "CV", "SEGMENT": "Upto 2.5 GVW", "INSURER": "Reliance, SBI, Tata", "PO": "-2%", "REMARKS": "NIL"},
    {"LOB": "CV", "SEGMENT": "All GVW & PCV 3W, GCV 3W", "INSURER": "Rest of Companies", "PO": "-2%", "REMARKS": "Payin Below 20%"},
    {"LOB": "CV", "SEGMENT": "All GVW & PCV 3W, GCV 3W", "INSURER": "Rest of Companies", "PO": "-3%", "REMARKS": "Payin 21% to 30%"},
    {"LOB": "CV", "SEGMENT": "All GVW & PCV 3W, GCV 3W", "INSURER": "Rest of Companies", "PO": "-4%", "REMARKS": "Payin 31% to 50%"},
    {"LOB": "CV", "SEGMENT": "All GVW & PCV 3W, GCV 3W", "INSURER": "Rest of Companies", "PO": "-5%", "REMARKS": "Payin Above 50%"},
    {"LOB": "BUS", "SEGMENT": "SCHOOL BUS", "INSURER": "TATA, Reliance, Digit, ICICI", "PO": "Less 2% of Payin", "REMARKS": "NIL"},
    {"LOB": "BUS", "SEGMENT": "SCHOOL BUS", "INSURER": "Rest of Companies", "PO": "88% of Payin", "REMARKS": "NIL"},
    {"LOB": "BUS", "SEGMENT": "STAFF BUS", "INSURER": "All Companies", "PO": "88% of Payin", "REMARKS": "NIL"},
    {"LOB": "TAXI", "SEGMENT": "TAXI", "INSURER": "All Companies", "PO": "-2%", "REMARKS": "Payin Below 20%"},
    {"LOB": "TAXI", "SEGMENT": "TAXI", "INSURER": "All Companies", "PO": "-3%", "REMARKS": "Payin 21% to 30%"},
    {"LOB": "TAXI", "SEGMENT": "TAXI", "INSURER": "All Companies", "PO": "-4%", "REMARKS": "Payin 31% to 50%"},
    {"LOB": "TAXI", "SEGMENT": "TAXI", "INSURER": "All Companies", "PO": "-5%", "REMARKS": "Payin Above 50%"},
    {"LOB": "MISD", "SEGMENT": "Misd, Tractor", "INSURER": "All Companies", "PO": "88% of Payin", "REMARKS": "NIL"}
]

def extract_text_from_file(file_bytes: bytes, filename: str, content_type: str) -> str:
    """Extract text from uploaded image file using OCR"""
    try:
        logger.info(f"📸 Extracting text from: {filename}")
        file_extension = filename.split('.')[-1].lower() if '.' in filename else ''
        
        image_extensions = ['png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff']
        if file_extension not in image_extensions and not (content_type and content_type.startswith('image/')):
            raise ValueError(f"Unsupported file type: {filename}")
        
        image_base64 = base64.b64encode(file_bytes).decode('utf-8')
        
        prompt = """The data is in a tabular format in the provided image. Extract ALL rows and ALL columns from the table.

Extract the data in JSON format where each object represents a row with the following key-value pairs:
- "Segment": The GVW or any  vehicle category (e.g., "0T-3.5T", "3.5T-7.5T", "45T Plus")
- "Policy Type": The policy type (e.g., "COMP/TP", use "COMP/TP" if not specified)
- "Location": The region and state code (e.g., "East:CG", "West:MH")
- "Payin": The payin value as a percentage (e.g., "23%", convert decimals like 0.625 to 62.5%)
- "Doable District": District information (use "N/A" if not found)
- "Remarks": Additional information (use empty string "" if none)

CRITICAL INSTRUCTIONS:
1. Extract EVERY row from the table.
2. Ignore columns named "Discount" or "CD1".
3. Return a valid JSON array.

Example: [{"Segment": "0T-3.5T", "Policy Type": "COMP/TP", "Location": "East:CG", "Payin": "34%", "Doable District": "N/A", "Remarks": ""}]
"""
              
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/{file_extension};base64,{image_base64}"}}
                ]
            }],
            temperature=0.0,
            max_tokens=16000
        )
        
        extracted_text = response.choices[0].message.content.strip()
        
        if not extracted_text or len(extracted_text) < 10:
            logger.error("❌ OCR returned very short or empty text")
            raise ValueError("OCR extraction failed - no text found")
        
        logger.info(f"✅ Extracted text length: {len(extracted_text)} characters")
        return extracted_text
        
    except Exception as e:
        logger.error(f"❌ Error in OCR extraction: {str(e)}")
        raise

def clean_json_response(response_text: str) -> str:
    """Clean and extract valid JSON array from OpenAI response"""
    cleaned = re.sub(r'```json\s*|\s*```', '', response_text).strip()
    
    start_idx = cleaned.find('[')
    end_idx = cleaned.rfind(']') + 1 if cleaned.rfind(']') != -1 else len(cleaned)
    
    if start_idx != -1 and end_idx > start_idx:
        cleaned = cleaned[start_idx:end_idx]
    else:
        logger.warning("⚠️ No valid JSON array found in response")
        return "[]"
    
    if not cleaned.startswith('['):
        cleaned = '[' + cleaned
    if not cleaned.endswith(']'):
        cleaned += ']'
    
    return cleaned

def ensure_list_format(data) -> list:
    """Ensure data is in list format"""
    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        return [data]
    else:
        raise ValueError(f"Expected list or dict, got {type(data)}")

def classify_payin(payin_str):
    """Converts Payin string to float and classifies its range"""
    try:
        payin_clean = str(payin_str).replace('%', '').replace(' ', '').strip()
        
        if not payin_clean or payin_clean.upper() == 'N/A':
            return 0.0, "Payin Below 20%"
        
        payin_value = float(payin_clean)
        
        if payin_value <= 20:
            category = "Payin Below 20%"
        elif 21 <= payin_value <= 30:
            category = "Payin 21% to 30%"
        elif 31 <= payin_value <= 50:
            category = "Payin 31% to 50%"
        else:
            category = "Payin Above 50%"
        return payin_value, category
    except (ValueError, TypeError) as e:
        logger.warning(f"⚠️ Could not parse payin value: {payin_str}, error: {e}")
        return 0.0, "Payin Below 20%"

def apply_formula_directly(policy_data, company_name):
    """Apply formula rules directly using Python logic"""
    if not policy_data:
        logger.warning("⚠️ No policy data to process")
        return []
    
    calculated_data = []
    
    for record in policy_data:
        try:
            segment = str(record.get('Segment', '')).upper()
            payin_value = record.get('Payin_Value', 0)
            payin_category = record.get('Payin_Category', '')
            
            lob = ""
            segment_upper = segment.upper()
            
            if any(tw_keyword in segment_upper for tw_keyword in ['TW', '2W', 'TWO WHEELER', 'TWO-WHEELER']):
                lob = "TW"
            elif any(car_keyword in segment_upper for car_keyword in ['PVT CAR', 'PRIVATE CAR', 'CAR', 'PVTCAR']):
                lob = "PVT CAR"
            elif any(cv_keyword in segment_upper for cv_keyword in ['CV', 'COMMERCIAL', 'LCV', 'GVW', 'TN', 'GCW', '0T-3.5T', '3.5T-7.5T', '7.5T-12T', '12T-20T', '20T-45T', '45T PLUS']):
                lob = "CV"
            elif 'BUS' in segment_upper:
                lob = "BUS"
            elif 'TAXI' in segment_upper:
                lob = "TAXI"
            elif any(misd_keyword in segment_upper for misd_keyword in ['MISD', 'TRACTOR', 'MISC']):
                lob = "MISD"
            else:
                remarks_upper = str(record.get('Remarks', '')).upper()
                if any(cv_keyword in remarks_upper for cv_keyword in ['TATA', 'MARUTI', 'GVW', 'TN']):
                    lob = "CV"
                else:
                    lob = "UNKNOWN"
            
            matched_segment = segment_upper
            if lob == "BUS":
                if "SCHOOL" not in segment_upper and "STAFF" not in segment_upper:
                    matched_segment = "STAFF BUS"
                elif "SCHOOL" in segment_upper:
                    matched_segment = "SCHOOL BUS"
                elif "STAFF" in segment_upper:
                    matched_segment = "STAFF BUS"
            
            if lob == "CV":
                if any(keyword in segment_upper for keyword in ["UPTO 2.5", "2.5 TN", "2.5 GVW", "2.5TN", "2.5GVW", "UPTO2.5"]):
                    matched_segment = "UPTO 2.5 GVW"
                else:
                    matched_segment = "ALL GVW & PCV 3W, GCV 3W"
            
            matched_rule = None
            rule_explanation = ""
            company_normalized = company_name.upper().replace('GENERAL', '').replace('INSURANCE', '').strip()
            
            for rule in FORMULA_DATA:
                if rule["LOB"] != lob:
                    continue
                    
                rule_segment = rule["SEGMENT"].upper()
                segment_match = False
                
                if lob == "CV":
                    if rule_segment == matched_segment:
                        segment_match = True
                elif lob == "BUS":
                    if matched_segment == rule_segment:
                        segment_match = True
                elif lob == "PVT CAR":
                    if "COMP" in rule_segment and any(keyword in segment for keyword in ["COMP", "COMPREHENSIVE"]):
                        segment_match = True
                    elif "TP" in rule_segment and "TP" in segment and "COMP" not in segment:
                        segment_match = True
                elif lob == "TW":
                    if "1+5" in rule_segment and "1+5" in segment:
                        segment_match = True
                    elif "SAOD + COMP" in rule_segment and any(keyword in segment for keyword in ["SAOD", "COMP"]):
                        segment_match = True
                    elif "TP" in rule_segment and "TP" in segment:
                        segment_match = True
                else:
                    segment_match = True
                
                if not segment_match:
                    continue
                
                insurers = [ins.strip().upper() for ins in rule["INSURER"].split(',')]
                company_match = False
                
                if "ALL COMPANIES" in insurers:
                    company_match = True
                elif "REST OF COMPANIES" in insurers:
                    is_in_specific_list = False
                    for other_rule in FORMULA_DATA:
                        if (other_rule["LOB"] == rule["LOB"] and 
                            other_rule["SEGMENT"] == rule["SEGMENT"] and
                            "REST OF COMPANIES" not in other_rule["INSURER"] and
                            "ALL COMPANIES" not in other_rule["INSURER"]):
                            other_insurers = [ins.strip().upper() for ins in other_rule["INSURER"].split(',')]
                            if any(company_key in company_normalized for company_key in other_insurers):
                                is_in_specific_list = True
                                break
                    if not is_in_specific_list:
                        company_match = True
                else:
                    for insurer in insurers:
                        if insurer in company_normalized or company_normalized in insurer:
                            company_match = True
                            break
                
                if not company_match:
                    continue
                
                remarks = rule.get("REMARKS", "")
                
                if remarks == "NIL" or "NIL" in remarks.upper():
                    matched_rule = rule
                    rule_explanation = f"Direct match: LOB={lob}, Segment={rule_segment}, Company={rule['INSURER']}"
                    break
                elif any(payin_keyword in remarks for payin_keyword in ["Payin Below", "Payin 21%", "Payin 31%", "Payin Above"]):
                    if payin_category in remarks:
                        matched_rule = rule
                        rule_explanation = f"Payin category match: LOB={lob}, Segment={rule_segment}, Payin={payin_category}"
                        break
                else:
                    matched_rule = rule
                    rule_explanation = f"Other remarks match: LOB={lob}, Segment={rule_segment}, Remarks={remarks}"
                    break
            
            if matched_rule:
                po_formula = matched_rule["PO"]
                calculated_payout = payin_value
                
                if "90% of Payin" in po_formula:
                    calculated_payout *= 0.9
                elif "88% of Payin" in po_formula:
                    calculated_payout *= 0.88
                elif "Less 2% of Payin" in po_formula:
                    calculated_payout -= 2
                elif "-2%" in po_formula:
                    calculated_payout -= 2
                elif "-3%" in po_formula:
                    calculated_payout -= 3
                elif "-4%" in po_formula:
                    calculated_payout -= 4
                elif "-5%" in po_formula:
                    calculated_payout -= 5
                
                calculated_payout = max(0, calculated_payout)
                formula_used = po_formula
            else:
                calculated_payout = payin_value
                formula_used = "No matching rule found"
            
            result_record = record.copy()
            result_record['Calculated Payout'] = f"{calculated_payout:.2f}%"
            result_record['Formula Used'] = formula_used
            result_record['Rule Explanation'] = rule_explanation
            
            calculated_data.append(result_record)
            
        except Exception as e:
            logger.error(f"❌ Error processing record: {record}, error: {str(e)}")
            result_record = record.copy()
            result_record['Calculated Payout'] = "Error"
            result_record['Formula Used'] = "Error in calculation"
            result_record['Rule Explanation'] = f"Error: {str(e)}"
            calculated_data.append(result_record)
    
    return calculated_data

def process_files(policy_file_bytes: bytes, policy_filename: str, policy_content_type: str, company_name: str):
    """Main processing function"""
    try:
        logger.info("=" * 50)
        logger.info("🚀 Starting file processing...")
        logger.info(f"📁 File: {policy_filename}, Size: {len(policy_file_bytes)} bytes")
        
        # Extract text
        extracted_text = extract_text_from_file(policy_file_bytes, policy_filename, policy_content_type)

        if not extracted_text.strip():
            raise ValueError("No text could be extracted from image")

        # Parse with AI
        logger.info("🧠 Parsing policy data with AI...")
        
        prompt_parse = f"""
Analyze this insurance policy text and extract structured data.
CRITICAL INSTRUCTIONS:
1. ALWAYS return a valid JSON ARRAY (list) of objects, even if there's only one record
2. Each object must have these EXACT field names:
   - "Segment": LOB + policy type (e.g., "TW TP", "PVT CAR COMP", "CV upto 2.5 Tn")
   - "Location": location/region information (use "N/A" if not found)
   - "Policy Type": policy type details (use "COMP/TP" if not specified)
   - "Payin": percentage value (convert decimals: 0.625 → 62.5%, or keep as is: 34%)
   - "Doable District": district info (use "N/A" if not found)
   - "Remarks": additional info including vehicle makes, age, transaction type, validity

3. For Segment field:
   - Identify LOB: TW, PVT CAR, CV, BUS, TAXI, MISD
   - Add policy type: TP, COMP, SAOD, etc.
   - For CV: preserve tonnage (e.g., "CV upto 2.5 Tn")

4. For Payin field:
   - If you see decimals like 0.625, convert to 62.5%
   - If you see whole numbers like 34, add % to make 34%
   - If you see percentages, keep them as is
   - Use the value from the "PO" column or any column that indicates payout/payin
   - Do not use values from "Discount" column for Payin

  5. For Remarks field - extract ALL additional info:
   - Vehicle makes (Tata, Maruti, etc.) → "Vehicle Makes: Tata, Maruti"
   - Age info (>5 years, etc.) → "Age: >5 years"
   - Transaction type (New/Old/Renewal) → "Transaction: New"
   - Validity dates → "Validity till: [date]"
   - Decline RTO information (e.g., "Decline RTO: Dhar, Jhabua")
   - Combine with semicolons: "Vehicle Makes: Tata; Age: >5 years; Transaction: New"
 IGNORE these columns completely - DO NOT extract them:
   - Discount
   - CD1
   - Any column containing "discount" or "cd1" 
   - These are not needed for our analysis

   
NOTE:
- Taxi PCV comes under the category of Taxi
- Multiple columns are there which has payouts based on either policy type or fuel type , so consider that as payin
- PCV < 6 STR comes under Taxi
-PC means Private Car and STP = TP
- Kali Pilli or Kaali Pilli means Taxi and it comes under Taxi
- If in SGEMENT OF Private Car , SAOD mentinoned then it comes into PVT CAR COMP + SAOD segment , also same for COMP
Here is the training Data:
I am training you

Text to analyze:
{extracted_text}

"""
       
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a data extraction expert. Extract policy data as a JSON array. Always return valid JSON."
                },
                {"role": "user", "content": prompt_parse}
            ],
            temperature=0.0,
            max_tokens=16000
        )
        
        parsed_json = response.choices[0].message.content.strip()
        logger.info(f"📝 Raw parsing response length: {len(parsed_json)}")
        
        cleaned_json = clean_json_response(parsed_json)
        
        policy_data = json.loads(cleaned_json)
        policy_data = ensure_list_format(policy_data)
        
        if not policy_data or len(policy_data) == 0:
            raise ValueError("Parsed data is empty")

        logger.info(f"✅ Successfully parsed {len(policy_data)} records")

        # Classify payin
        for record in policy_data:
            if 'Discount' in record:
                del record['Discount']
            payin_val, payin_cat = classify_payin(record.get('Payin', '0%'))
            record['Payin_Value'] = payin_val
            record['Payin_Category'] = payin_cat

        # Apply formulas
        logger.info("🧮 Applying formulas...")
        calculated_data = apply_formula_directly(policy_data, company_name)
        
        if not calculated_data:
            raise ValueError("No data after formula application")

        logger.info(f"✅ Successfully calculated {len(calculated_data)} records")

        # Create Excel
        logger.info("📊 Creating Excel file...")
        df_calc = pd.DataFrame(calculated_data)
        
        if df_calc.empty:
            raise ValueError("DataFrame is empty")

        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_calc.to_excel(writer, sheet_name='Policy Data', startrow=2, index=False)
            worksheet = writer.sheets['Policy Data']
            headers = list(df_calc.columns)
            for col_num, value in enumerate(headers, 1):
                cell = worksheet.cell(row=3, column=col_num, value=value)
                cell.font = cell.font.copy(bold=True)
            if len(headers) > 1:
                company_cell = worksheet.cell(row=1, column=1, value=company_name)
                worksheet.merge_cells(start_row=1, start_column=1, end_row=1, end_column=len(headers))
                company_cell.font = company_cell.font.copy(bold=True, size=14)
                company_cell.alignment = company_cell.alignment.copy(horizontal='center')
                title_cell = worksheet.cell(row=2, column=1, value='Policy Data with Payin and Calculated Payouts')
                worksheet.merge_cells(start_row=2, start_column=1, end_row=2, end_column=len(headers))
                title_cell.font = title_cell.font.copy(bold=True, size=12)
                title_cell.alignment = title_cell.alignment.copy(horizontal='center')

        output.seek(0)
        excel_data = output.read()
        excel_data_base64 = base64.b64encode(excel_data).decode('utf-8')

        # Calculate metrics
        avg_payin = sum([r.get('Payin_Value', 0) for r in calculated_data]) / len(calculated_data) if calculated_data else 0.0
        unique_segments = len(set([r.get('Segment', 'N/A') for r in calculated_data]))
        formula_summary = {}
        for record in calculated_data:
            formula = record.get('Formula Used', 'Unknown')
            formula_summary[formula] = formula_summary.get(formula, 0) + 1

        logger.info("✅ Processing completed successfully")
        logger.info("=" * 50)
        
        return {
            "extracted_text": extracted_text,
            "parsed_data": policy_data,
            "calculated_data": calculated_data,
            "excel_data": excel_data_base64,
            "csv_data": df_calc.to_csv(index=False),
            "formula_data": FORMULA_DATA,
            "avg_payin": round(avg_payin, 1),
            "unique_segments": unique_segments,
            "formula_summary": formula_summary
        }

    except Exception as e:
        logger.error(f"❌ Error in process_files: {str(e)}", exc_info=True)
        raise

@app.get("/")
async def root():
    """Serve the HTML frontend"""
    try:
        html_path = Path("index.html")
        if not html_path.exists():
            return HTMLResponse(
                content="<h1>Error: index.html not found</h1><p>Please ensure index.html is in the same directory as main.py</p>",
                status_code=404
            )
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except Exception as e:
        logger.error(f"❌ Error serving HTML: {str(e)}")
        return HTMLResponse(content=f"<h1>Error loading page</h1><p>{str(e)}</p>", status_code=500)

@app.post("/process")
async def process_policy(company_name: str = Form(...), policy_file: UploadFile = File(...)):
    """Process policy endpoint"""
    try:
        logger.info("=" * 50)
        logger.info(f"📨 Received request for company: {company_name}")
        logger.info(f"📄 File: {policy_file.filename}, Content-Type: {policy_file.content_type}")
        
        # Read file
        policy_file_bytes = await policy_file.read()
        if len(policy_file_bytes) == 0:
            logger.error("❌ Uploaded file is empty")
            return JSONResponse(
                status_code=400,
                content={"error": "Uploaded file is empty"}
            )

        logger.info(f"📦 File size: {len(policy_file_bytes)} bytes")
        
        # Process
        results = process_files(
            policy_file_bytes, 
            policy_file.filename, 
            policy_file.content_type,
            company_name
        )
        
        logger.info("✅ Returning results to client")
        return JSONResponse(content=results)
        
    except ValueError as e:
        logger.error(f"❌ Validation error: {str(e)}")
        return JSONResponse(
            status_code=400,
            content={"error": str(e)}
        )
    except Exception as e:
        logger.error(f"❌ Error processing request: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": f"Processing failed: {str(e)}"}
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return JSONResponse(content={"status": "healthy", "message": "Server is running"})

if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 Starting Insurance Policy Processing System...")
    logger.info("📡 Server will be available at: http://localhost:8000")
    logger.info("🔑 OpenAI API Key is configured: ✅")
    uvicorn.run(app, host="0.0.0.0", port=8000)
