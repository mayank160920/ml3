import json
from typing import List


MOCK_RESPONSES = {
    "Plan Overview": {
        "sbc": {
            "section": "Plan Overview",
            "source_pages": [1],
            "extractions": [
                {"entity_name": "Individual_Deductible_In_Network", "extracted_value": "$0", "status": "EXTRACTED", "source_page": 1, "source_region": "Important Questions table, row: What is the overall deductible?", "confidence": 0.99, "raw_context": "$0 individual/$0 family network."},
                {"entity_name": "Family_Deductible_In_Network", "extracted_value": "$0", "status": "EXTRACTED", "source_page": 1, "source_region": "Important Questions table, row: What is the overall deductible?", "confidence": 0.99, "raw_context": "$0 individual/$0 family network."},
                {"entity_name": "Individual_OOP_Max_In_Network", "extracted_value": "$7,500", "status": "EXTRACTED", "source_page": 1, "source_region": "Important Questions table, row: What is the out-of-pocket limit?", "confidence": 0.99, "raw_context": "$7,500 individual/$15,000 family network."},
                {"entity_name": "Family_OOP_Max_In_Network", "extracted_value": "$15,000", "status": "EXTRACTED", "source_page": 1, "source_region": "Important Questions table, row: What is the out-of-pocket limit?", "confidence": 0.99, "raw_context": "$7,500 individual/$15,000 family network."},
            ]
        },
        "bg": {
            "section": "Plan Overview",
            "source_pages": [1],
            "extractions": [
                {"entity_name": "Individual_Deductible_In_Network", "extracted_value": "$0", "status": "EXTRACTED", "source_page": 1, "source_region": "Plan Overview section, Deductible (Individual) row", "confidence": 0.97, "raw_context": "Deductible (Individual)  $0"},
                {"entity_name": "Family_Deductible_In_Network", "extracted_value": "$0", "status": "EXTRACTED", "source_page": 1, "source_region": "Plan Overview section, Deductible (Family) row", "confidence": 0.97, "raw_context": "Deductible (Family)  $0"},
                {"entity_name": "Individual_OOP_Max_In_Network", "extracted_value": "$7,500", "status": "EXTRACTED", "source_page": 1, "source_region": "Plan Overview section, Out-of-Pocket Maximum (Individual) row", "confidence": 0.97, "raw_context": "Out-of-Pocket Maximum (Individual)  $7,500"},
                {"entity_name": "Family_OOP_Max_In_Network", "extracted_value": "$15,000", "status": "EXTRACTED", "source_page": 1, "source_region": "Plan Overview section, Out-of-Pocket Maximum (Family) row", "confidence": 0.97, "raw_context": "Out -of-Pocket Maximum (Family)  $15,000"},
            ]
        }
    },
    "Doctor Visits": {
        "sbc": {
            "section": "Doctor Visits",
            "source_pages": [2],
            "extractions": [
                {"entity_name": "Primary_Care_Visit", "extracted_value": "$20 copay/visit", "status": "EXTRACTED", "source_page": 2, "source_region": "Common Medical Event table, Primary care visit row, Network Provider column", "confidence": 0.99, "raw_context": "Primary care visit to treat an injury or illness $20 copay/visit"},
                {"entity_name": "Specialist_Visit", "extracted_value": "$20 copay/visit", "status": "EXTRACTED", "source_page": 2, "source_region": "Common Medical Event table, Specialist visit row, Network Provider column", "confidence": 0.99, "raw_context": "Specialist visit $20 copay/visit"},
                {"entity_name": "Preventive_Care", "extracted_value": "No charge", "status": "EXTRACTED", "source_page": 2, "source_region": "Common Medical Event table, Preventive care row, Network Provider column", "confidence": 0.99, "raw_context": "Preventive care/screening/Immunization No charge; deductible does not apply."},
            ]
        },
        "bg": {
            "section": "Doctor Visits",
            "source_pages": [1],
            "extractions": [
                {"entity_name": "Primary_Care_Visit", "extracted_value": "$20 copay/visit", "status": "EXTRACTED", "source_page": 1, "source_region": "Doctor Visits section, Primary Care Visit row", "confidence": 0.96, "raw_context": "Doctor Visits  Primary Care Visit  $20 copay/visit"},
                {"entity_name": "Specialist_Visit", "extracted_value": "$20 - copay/visit", "status": "EXTRACTED", "source_page": 1, "source_region": "Doctor Visits section, Specialist Visit row", "confidence": 0.95, "raw_context": "Doctor Visits  Specialist Visit  $20 - copay/visit"},
                {"entity_name": "Preventive_Care", "extracted_value": "No charge", "status": "EXTRACTED", "source_page": 1, "source_region": "Preventive Services section, Preventive Care row", "confidence": 0.97, "raw_context": "Preventive Services  Preventive Care / Screening  No charge"},
            ]
        }
    },
    "Medical Tests": {
        "sbc": {
            "section": "Medical Tests",
            "source_pages": [2],
            "extractions": [
                {"entity_name": "Diagnostic_Test", "extracted_value": "$35 copay/visit", "status": "EXTRACTED", "source_page": 2, "source_region": "Common Medical Event table, Diagnostic test row, Network Provider column", "confidence": 0.99, "raw_context": "Diagnostic test (x-ray, blood work) $35 copay/visit"},
                {"entity_name": "Imaging", "extracted_value": "$400 copay/visit", "status": "EXTRACTED", "source_page": 2, "source_region": "Common Medical Event table, Imaging row, Network Provider column", "confidence": 0.99, "raw_context": "Imaging (CT/PET scans, MRIs) $400 copay/visit"},
            ]
        },
        "bg": {
            "section": "Medical Tests",
            "source_pages": [1],
            "extractions": [
                {"entity_name": "Diagnostic_Test", "extracted_value": "$35 copay/visit", "status": "EXTRACTED", "source_page": 1, "source_region": "Medical Tests section, Diagnostic Test row", "confidence": 0.96, "raw_context": "Medical Tests  Diagnostic Test (X-ray / Lab Work)  $35 copay/visit"},
                {"entity_name": "Imaging", "extracted_value": "$400 copay/visit", "status": "EXTRACTED", "source_page": 1, "source_region": "Medical Tests section, Imaging row", "confidence": 0.96, "raw_context": "Medical Tests  Imaging (CT / MRI / PET Scan)  $400 copay/visit"},
            ]
        }
    },
    "Emergency Care": {
        "sbc": {
            "section": "Emergency Care",
            "source_pages": [3],
            "extractions": [
                {"entity_name": "Emergency_Room", "extracted_value": "$300 copay/visit", "status": "EXTRACTED", "source_page": 3, "source_region": "Common Medical Event table, Emergency room care row, Network Provider column", "confidence": 0.99, "raw_context": "Emergency room care $300 copay/visit"},
                {"entity_name": "Emergency_Medical_Transport", "extracted_value": "30% coinsurance", "status": "EXTRACTED", "source_page": 3, "source_region": "Common Medical Event table, Emergency medical transportation row, Network Provider column", "confidence": 0.99, "raw_context": "Emergency medical transportation 30% coinsurance"},
                {"entity_name": "Urgent_Care", "extracted_value": "$40 copay/visit", "status": "EXTRACTED", "source_page": 3, "source_region": "Common Medical Event table, Urgent care row, Network Provider column", "confidence": 0.99, "raw_context": "Urgent care $40 copay/visit"},
            ]
        },
        "bg": {
            "section": "Emergency Care",
            "source_pages": [1],
            "extractions": [
                {"entity_name": "Emergency_Room", "extracted_value": "$300 copay/visit", "status": "EXTRACTED", "source_page": 1, "source_region": "Emergency Care section, Emergency Room Care row", "confidence": 0.97, "raw_context": "Emergency Care  Emergency Room Care  $300 copay/visit"},
                {"entity_name": "Emergency_Medical_Transport", "extracted_value": "30% coinsurance", "status": "EXTRACTED", "source_page": 1, "source_region": "Emergency Care section, Emergency Medical Transport row", "confidence": 0.96, "raw_context": "Emergency Care  Emergency Medical Transport  30% coinsurance"},
                {"entity_name": "Urgent_Care", "extracted_value": "$40 copay/visit", "status": "EXTRACTED", "source_page": 1, "source_region": "Emergency Care section, Urgent Care row", "confidence": 0.97, "raw_context": "Emergency Care  Urgent Care  $40 copay/visit"},
            ]
        }
    },
    "Hospital Services": {
        "sbc": {
            "section": "Hospital Services",
            "source_pages": [3],
            "extractions": [
                {"entity_name": "Hospital_Facility_Fee", "extracted_value": "$500 copay/visit", "status": "EXTRACTED", "source_page": 3, "source_region": "Common Medical Event table, Facility fee row, Network Provider column", "confidence": 0.99, "raw_context": "Facility fee (e.g., hospital room) $500 copay/visit"},
                {"entity_name": "Hospital_Physician_Fee", "extracted_value": "30% coinsurance", "status": "EXTRACTED", "source_page": 3, "source_region": "Common Medical Event table, Physician/surgeon fees row, Network Provider column", "confidence": 0.99, "raw_context": "Physician/surgeon fees 30% coinsurance"},
            ]
        },
        "bg": {
            "section": "Hospital Services",
            "source_pages": [1],
            "extractions": [
                {"entity_name": "Hospital_Facility_Fee", "extracted_value": "$500 copay/visit", "status": "EXTRACTED", "source_page": 1, "source_region": "Hospital Services section, Hospital Stay (Facility Fee) row", "confidence": 0.97, "raw_context": "Hospital Services  Hospital Stay (Facility Fee)  $500 copay/visit"},
                {"entity_name": "Hospital_Physician_Fee", "extracted_value": "30% coinsurance", "status": "EXTRACTED", "source_page": 1, "source_region": "Hospital Services section, Physician / Surgeon Fees row", "confidence": 0.96, "raw_context": "Hospital Services  Physician / Surgeon Fees  30% coinsurance"},
            ]
        }
    },
    "Prescription Drugs": {
        "sbc": {
            "section": "Prescription Drugs",
            "source_pages": [2],
            "extractions": [
                {"entity_name": "Drug_Tier_1_Generic", "extracted_value": "No charge", "status": "EXTRACTED", "source_page": 2, "source_region": "Common Medical Event table, Tier 1 row, Network Provider column", "confidence": 0.99, "raw_context": "Tier 1 No charge per prescription (retail)"},
                {"entity_name": "Drug_Tier_2_Preferred_Brand", "extracted_value": "$30/$60/$90 copay", "status": "EXTRACTED", "source_page": 2, "source_region": "Common Medical Event table, Tier 2 row, Network Provider column", "confidence": 0.99, "raw_context": "Tier 2 $30/$60/$90 copay per prescription (retail)"},
                {"entity_name": "Drug_Tier_3_Nonpreferred_Brand", "extracted_value": "$150/$300/$450 copay", "status": "EXTRACTED", "source_page": 2, "source_region": "Common Medical Event table, Tier 3 row, Network Provider column", "confidence": 0.99, "raw_context": "Tier 3 $150/$300/$450 copay per prescription (retail)"},
            ]
        },
        "bg": {
            "section": "Prescription Drugs",
            "source_pages": [1],
            "extractions": [
                {"entity_name": "Drug_Tier_1_Generic", "extracted_value": "No charge per prescription", "status": "EXTRACTED", "source_page": 1, "source_region": "Prescription Drugs section, Generic Drugs (Tier 1) row", "confidence": 0.96, "raw_context": "Prescription Drugs  Generic Drugs (Tier 1)  No charge per prescription"},
                {"entity_name": "Drug_Tier_2_Preferred_Brand", "extracted_value": "$30/$60/$90 copay per prescription", "status": "EXTRACTED", "source_page": 1, "source_region": "Prescription Drugs section, Preferred Brand Drugs (Tier 2) row", "confidence": 0.96, "raw_context": "Prescription Drugs  Preferred Brand Drugs (Tier 2)  $30/$60/$90 copay per prescription"},
                {"entity_name": "Drug_Tier_3_Nonpreferred_Brand", "extracted_value": "$150/$300/$450 copay per prescription", "status": "EXTRACTED", "source_page": 1, "source_region": "Prescription Drugs section, Non-Preferred Brand Drugs (Tier 3) row", "confidence": 0.96, "raw_context": "Prescription Drugs  Non-Preferred Brand Drugs (Tier 3)  $150/$300/$450 copay per prescription"},
            ]
        }
    },
    "Rehabilitation": {
        "sbc": {
            "section": "Rehabilitation",
            "source_pages": [5],
            "extractions": [
                {"entity_name": "Rehabilitation_Services", "extracted_value": "$45 copay/visit", "status": "EXTRACTED", "source_page": 5, "source_region": "Common Medical Event table, Rehabilitation services row, Network Provider column", "confidence": 0.99, "raw_context": "Rehabilitation services $45 copay/visit"},
                {"entity_name": "Durable_Medical_Equipment", "extracted_value": "30% coinsurance", "status": "EXTRACTED", "source_page": 5, "source_region": "Common Medical Event table, Durable medical equipment row, Network Provider column", "confidence": 0.99, "raw_context": "Durable medical equipment 30% coinsurance"},
            ]
        },
        "bg": {
            "section": "Rehabilitation",
            "source_pages": [1],
            "extractions": [
                {"entity_name": "Rehabilitation_Services", "extracted_value": "$45 copay/visit", "status": "EXTRACTED", "source_page": 1, "source_region": "Rehabilitation Services section, Physical Therapy row", "confidence": 0.96, "raw_context": "Rehabilitation Services  Physical Therapy  $45 copay/visit"},
                {"entity_name": "Durable_Medical_Equipment", "extracted_value": "30 % coinsurance", "status": "EXTRACTED", "source_page": 1, "source_region": "Equipment section, Durable Medical Equipment row", "confidence": 0.95, "raw_context": "Equipment  Durable Medical Equipment  30 % coinsurance"},
            ]
        }
    },
    "Mental Health": {
        "sbc": {
            "section": "Mental Health",
            "source_pages": [3],
            "extractions": [
                {"entity_name": "Mental_Health_Outpatient", "extracted_value": "$20 copay/visit", "status": "EXTRACTED", "source_page": 3, "source_region": "Common Medical Event table, Outpatient services row, Network Provider column", "confidence": 0.99, "raw_context": "Outpatient services $20 copay/visit"},
                {"entity_name": "Mental_Health_Inpatient", "extracted_value": "$500 copay/visit", "status": "EXTRACTED", "source_page": 3, "source_region": "Common Medical Event table, Inpatient services row, Network Provider column", "confidence": 0.99, "raw_context": "Inpatient services $500 copay/visit"},
            ]
        },
        "bg": {
            "section": "Mental Health",
            "source_pages": [1],
            "extractions": [
                {"entity_name": "Mental_Health_Outpatient", "extracted_value": "$20 copay/visit", "status": "EXTRACTED", "source_page": 1, "source_region": "Mental Health section, Mental Health Outpatient Services row", "confidence": 0.97, "raw_context": "Mental Health  Mental Health Outpatient Services  $20 copay/visit"},
                {"entity_name": "Mental_Health_Inpatient", "extracted_value": "$500 copay/visit", "status": "EXTRACTED", "source_page": 1, "source_region": "Mental Health section, Mental Health Inpatient Services row", "confidence": 0.97, "raw_context": "Mental Health  Mental Health Inpatient Services  $500 copay/visit"},
            ]
        }
    },
}


class MockMLLMClient:

    def __init__(self, document_type: str = "sbc"):
        if document_type not in ("sbc", "bg"):
            raise ValueError("document_type must be 'sbc' or 'bg'")
        self.document_type = document_type

    def generate(self, prompt: str, page_images: list) -> str:
        section_name = None
        for line in prompt.split("\n"):
            if line.startswith("Section:"):
                section_name = line.replace("Section:", "").strip()
                break

        if section_name and section_name in MOCK_RESPONSES:
            response = MOCK_RESPONSES[section_name][self.document_type]
            return json.dumps(response)

        fallback = {
            "section": section_name or "Unknown",
            "source_pages": [],
            "extractions": []
        }
        return json.dumps(fallback)
