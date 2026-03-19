from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
from openai import OpenAI
import os, json, re, httpx
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

class PatientInfo(BaseModel):
    full_name: Optional[str] = None
    date_of_birth: Optional[str] = None

    @field_validator("date_of_birth", mode="before")
    @classmethod
    def validate_dob(cls, v):
        if v is None:
            return v
        for fmt in ["%Y-%m-%d", "%m/%d/%Y", "%m-%d-%Y", "%m/%d/%y"]:
            try:
                dt = datetime.strptime(v, fmt)
                return dt.date().isoformat()
            except ValueError:
                continue
        try:
            dt = datetime.fromisoformat(v)
            return dt.date().isoformat()
        except Exception:
            raise ValueError("date_of_birth must be a valid date in YYYY-MM-DD or MM/DD/YYYY format")

class InsuranceInfo(BaseModel):
    payer_name: Optional[str] = None
    insurance_id: Optional[str] = None

class MedicalInfo(BaseModel):
    chief_complaint: Optional[str] = None

class AddressInfo(BaseModel):
    street: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None
    formatted_address: Optional[str] = None


US_STATE_CODES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
    "DC"
}

class AppointmentSlot(BaseModel):
    provider_id: int
    provider_name: str
    specialty: str
    datetime: datetime
    available: bool = True

class CollectedData(BaseModel):
    patient: PatientInfo = Field(default_factory=PatientInfo)
    insurance: InsuranceInfo = Field(default_factory=InsuranceInfo)
    medical: MedicalInfo = Field(default_factory=MedicalInfo)
    address: AddressInfo = Field(default_factory=AddressInfo)
    selected_appointment: Optional[AppointmentSlot] = None
    appointment_confirmed: bool = False

def generate_mock_appointments() -> List[AppointmentSlot]:
    providers = [
        {"id": 1, "name": "Dr. Sarah Johnson", "specialty": "Family Medicine"},
        {"id": 2, "name": "Dr. Michael Chen", "specialty": "Internal Medicine"},
        {"id": 3, "name": "Dr. Emily Rodriguez", "specialty": "Pediatrics"}
    ]

    appointments = []
    base_date = datetime.now()
    for day in range(7):
        date = base_date + timedelta(days=day)
        for hour in [9,10,11,1,2,3]:
            for provider in providers:
                appointments.append(
                    AppointmentSlot(
                        provider_id = provider['id'],
                        provider_name = provider['name'],
                        specialty = provider['specialty'],
                        datetime = date.replace(hour=hour, minute = 0, second=0, microsecond=0)
                    )
            )
    return appointments



"""
Need to collect one piece of information at a time/go step by step.
Steps for collecting patient information:
1. Ask for full name
2. Ask for date of birth
3. Ask for insurance information (payer name)
4. Ask for chief complaint
5. Ask for address
6. Ask to select an appointment slot
"""
class AppointmentAgent:
    def __init__(
        self, 
        openai_api_key: str, 
        appointments: List[AppointmentSlot],
        google_maps_api_key: Optional[str] = None,
        max_history: int = 10
    ):
        self.client = OpenAI(api_key=openai_api_key)
        self.http_client = httpx.Client(timeout= 10.0)
        self.google_maps_api_key = google_maps_api_key
        self.all_appointments = appointments
        self.collected_data = CollectedData()
        self.current_step = "greeting"
        self.last_displayed = {}
        self.system_prompt = """
        You are a professional medical appointment scheduling assistant.
        Your only tasks are to collect, one at a time, ONLY the following information:
        1. Full name
        2. Date of birth
        3. Insurance payer name
        4. Chief complaint (reason for visit)
        5. Address (street, city, state, zip)
        6. Appointment slot selection
        NEVER ask for email address, phone number, social security number, or any other details not listed. Be friendly and empathetic.
        Current step: {current_step}. Collected data so far: {collected_data}.
        """
        self.conversation_history = []
        self.max_history = max_history
        self.extraction_function = {
            "name": "extract_fields",
            "description": "Extract structured appointment intake fields from a user message.",
            "parameters": {
                "type": "object",
                "properties": {
                    "full_name": {"type": "string"},
                    "date_of_birth": {"type": "string", "description": "YYYY-MM-DD or MM/DD/YYYY"},
                    "payer_name": {"type": "string"},
                    "chief_complaint": {"type": "string"},
                    "street": {"type": "string"},
                    "city": {"type": "string"},
                    "state": {"type": "string"},
                    "zip_code": {"type": "string"},
                }
            }
        }


    def add_to_history(self, role: str, content:str):
        self.conversation_history.append({"role":role, "content":content})
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
    
    def extract(self, user_input:str):
        extracted = self.fallback_extract(user_input)
        
        try:
            response = self.client.responses.create(
                model = "gpt-4o-mini",
                input = [
                    {"role": "system", "content": "Extract structured fields from user message."},
                    {"role": "user", "content": user_input}
                ],
                tools = [{"type": "function", "function": self.extraction_function}],
                tool_choice = {"type": "function", "function": {"name": "extract_fields"}},
                temperature = 0
            )

            if response.output and response.output[0].type == "function_call":
                func = response.output[0]
                args = func.arguments

                if args:
                    if isinstance(args, str):
                        try:
                            parsed = json.loads(args)
                            if isinstance(parsed, dict):
                                extracted.update({k: v for k, v in parsed.items() if v})
                        except Exception:
                            pass
                    elif isinstance(args, dict):
                        extracted.update({k: v for k, v in args.items() if v})
            if response.output_text:
                try:
                    parsed = json.loads(response.output_text)
                    if isinstance(parsed, dict):
                        extracted.update({k: v for k, v in parsed.items() if v})
                except Exception:
                    pass
        except Exception:
            try:
                output_text = response.output_text
            except Exception:
                return extracted
            start = output_text.find("{")
            end = output_text.find("}")
            if start == -1 or end == -1:
                return extracted
            try:
                parsed = json.loads(output_text[start:end+1])
                if isinstance(parsed, dict):
                    extracted.update({k: v for k, v in parsed.items() if v})
            except Exception:
                return extracted
        return extracted

    def fallback_extract(self, user_input: str):
        extracted = {}
        text = user_input.strip()

        if self.current_step == "full_name" and text:
            extracted["full_name"] = text
        elif self.current_step == "date_of_birth" and text:
            extracted["date_of_birth"] = text
        elif self.current_step == "payer_name" and text:
            extracted["payer_name"] = text
        elif self.current_step == "chief_complaint" and text:
            extracted["chief_complaint"] = text
        elif self.current_step == "address":
            parts = [part.strip() for part in text.split(",") if part.strip()]
            if len(parts) >= 4:
                extracted["street"] = parts[0]
                extracted["city"] = parts[1]
                extracted["state"] = parts[2]
                extracted["zip_code"] = parts[3]
            else:
                match = re.match(
                    r"^(?P<street>.+?),\s*(?P<city>[^,]+?),\s*(?P<state>[A-Za-z]{2})\s+(?P<zip_code>\d{5}(?:-\d{4})?)$",
                    text
                )
                if match:
                    extracted.update(match.groupdict())
        return extracted
            

    def apply_extracted(self, extracted):
        if not isinstance(extracted, dict):
            return None
        if "full_name" in extracted:
            self.collected_data.patient.full_name = extracted["full_name"]
        if "date_of_birth" in extracted:
            try:
                validated_patient = PatientInfo(date_of_birth=extracted["date_of_birth"])
                self.collected_data.patient.date_of_birth = validated_patient.date_of_birth
            except ValueError as exc:
                return str(exc)
        if "payer_name" in extracted:
            self.collected_data.insurance.payer_name = extracted["payer_name"]
        if "chief_complaint" in extracted:
            self.collected_data.medical.chief_complaint = extracted["chief_complaint"]
        if "street" in extracted:
            self.collected_data.address.street = extracted["street"]
        if "city" in extracted:
            self.collected_data.address.city = extracted["city"]
        if "state" in extracted:
            self.collected_data.address.state = extracted["state"]
        if "zip_code" in extracted:
            self.collected_data.address.zip_code = extracted["zip_code"]
        return None

    def prompt_for_step(self, step: str) -> str:
        prompts = {
            "full_name": "Could you please tell me your full name?",
            "date_of_birth": "What is your date of birth? You can use YYYY-MM-DD, MM/DD/YYYY, MM-DD-YYYY, or MM/DD/YY.",
            "payer_name": "What is your insurance payer name?",
            "chief_complaint": "What is the reason for your visit?",
            "address": "What is your address? Please use `street, city, state, zip` or `street, city, state zip`.",
            "appointment_selection": "I have appointment options ready. Please choose one by number.",
            "confirmation": "Please reply yes to confirm this appointment or no to see other options."
        }
        return prompts.get(step, "Could you provide the next detail?")

    def generate_assistant_reply(self, user_input: str) -> str:
        # Build context for the assistant with current step and collected data
        context = self.system_prompt.format(
            current_step = self.current_step,
            collected_data = self.collected_data.model_dump_json(indent = 2, exclude_none=True)
        )
        messages = [{"role": "system", "content": context}] + self.conversation_history

        try:
            response = self.client.responses.create(
                model = 'gpt-4o-mini',
                input = messages + [{"role": "user", "content": user_input}],
                temperature = .7
            )
            if response.output_text:
                return response.output_text.strip()
        except Exception:
            return "Thanks. Could you provide the next detail?"

    def get_next_step(self) -> str:
        if not self.collected_data.patient.full_name:
            return "full_name"
        if not self.collected_data.patient.date_of_birth:
            return "date_of_birth"
        if not self.collected_data.insurance.payer_name:
            return "payer_name"
        if not self.collected_data.medical.chief_complaint:
            return "chief_complaint"
        if not self.collected_data.address.formatted_address:
            return "address"
        if not self.collected_data.selected_appointment:
            return "appointment_selection"
        if not self.collected_data.appointment_confirmed:
            return "confirmation"
        return "completed"

    def validate_address_locally(self, address_info: AddressInfo):
        street = (address_info.street or "").strip()
        city = (address_info.city or "").strip()
        state = (address_info.state or "").strip().upper()
        zip_code = (address_info.zip_code or "").strip()

        if not all([street, city, state, zip_code]):
            return False, None, None, "missing required address fields"

        if not re.match(r"^\d+\s+.+", street):
            return False, None, None, "street should start with a street number"

        if not re.match(r"^[A-Za-z .'-]+$", city):
            return False, None, None, "city contains unexpected characters"

        if state not in US_STATE_CODES:
            return False, None, None, "state must be a valid 2-letter US code"

        if not re.match(r"^\d{5}(?:-\d{4})?$", zip_code):
            return False, None, None, "zip code must be 5 digits or ZIP+4"

        formatted_address = f"{street}, {city}, {state} {zip_code}"
        components = {
            "city": city,
            "state": state,
            "zip_code": zip_code
        }
        street_match = re.match(r"^(?P<street_number>\d+)\s+(?P<route>.+)$", street)
        if street_match:
            components.update(street_match.groupdict())

        return True, formatted_address, components, None

    def validate_address_with_google(self, address_info: AddressInfo):
        if not self.google_maps_api_key:
            return None, None, None, "GOOGLE_MAPS_API_KEY not configured"

        address = ", ".join(
            part for part in [
                address_info.street,
                address_info.city,
                address_info.state,
                address_info.zip_code,
            ]
            if part
        )

        try:
            response = self.http_client.get(
                "https://maps.googleapis.com/maps/api/geocode/json",
                params={"address": address, "key": self.google_maps_api_key},
            )
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            return None, None, None, str(exc)

        status = data.get("status")
        if status == "OK" and data.get("results"):
            result = data["results"][0]
            formatted_address = result.get("formatted_address")
            components = {}
            for component in result.get("address_components", []):
                types = component.get("types", [])
                if "street_number" in types:
                    components["street_number"] = component.get("long_name")
                if "route" in types:
                    components["route"] = component.get("long_name")
                if "locality" in types:
                    components["city"] = component.get("long_name")
                if "administrative_area_level_1" in types:
                    components["state"] = component.get("short_name")
                if "postal_code" in types:
                    components["zip_code"] = component.get("long_name")
            return True, formatted_address, components, None

        return False, None, None, status

    def validate_address(self, address_info: AddressInfo):
        google_valid, google_formatted, google_components, google_error = self.validate_address_with_google(address_info)
        if google_valid:
            return True, google_formatted, google_components, None
        if google_error == "REQUEST_DENIED":
            local_valid, local_formatted, local_components, local_error = self.validate_address_locally(address_info)
            detail = "Google Maps denied the request; used local validation instead."
            if local_error:
                detail = f"{detail} Local validation detail: {local_error}."
            return local_valid, local_formatted, local_components, detail
        if google_valid is None:
            return self.validate_address_locally(address_info)
        return False, None, None, google_error

    def format_appointments(self, days: int = 3, per_day: int = 3):
        appointments_by_date = {}

        cnt = min(len(self.all_appointments), 21)
        for a in self.all_appointments[:cnt]:
            k = a.datetime.strftime("%A, %B %d, %Y")
            appointments_by_date.setdefault(k, []).append(a)
        
        output = ["\nAvailable Appointments:", ]
        idx = 1
        mapping = {}
        for date, apts in list(appointments_by_date.items())[:days]:
            output.append(f"\n{date}")
            step = max(1, len(apts) // per_day)
            sampled_apts = apts[::step][:per_day]
            for a in sampled_apts:
                t = a.datetime.strftime("%I:%M %p")
                output.append(f"  [{idx}] {t} - {a.provider_name} ({a.specialty})")
                mapping[idx] = a
                idx += 1
        self.last_displayed = mapping
        return "\n".join(output), mapping

    def get_appointment_by_number(self, number: int):
        return self.last_displayed.get(int(number))

    def is_affirmative(self, user_input: str) -> bool:
        return user_input.strip().lower() in {"yes", "y", "yeah", "yep", "sure", "ok", "okay", "correct"}

    def is_negative(self, user_input: str) -> bool:
        return user_input.strip().lower() in {"no", "n", "nope", "nah"}
    
    def display_confirmation(self):
        a = self.collected_data.selected_appointment
        if not a:
            return "No appointment selected."
        summary = (
            "APPOINTMENT CONFIRMATION\nDetails are below:\n"
            f"Date & Time: {a.datetime.strftime('%A, %B %d at %I:%M %p')}\n"
            f"Provider: {a.provider_name}\n"
            f"Specialty: {a.specialty}\n"
            f"Reason for visit: {self.collected_data.medical.chief_complaint}\n\n"
            "Thank you for scheduling with us!"
        )

        return summary

    def chat(self, user_input: str):
        # Add user message to conversation history
        self.add_to_history("user", user_input)
        self.current_step = self.get_next_step()

        if self.current_step == "confirmation":
            if self.is_affirmative(user_input):
                self.collected_data.appointment_confirmed = True
                self.current_step = self.get_next_step()
                text = "Perfect. Your appointment is confirmed."
                self.add_to_history("assistant", text)
                return {"type": "reply_and_ready_to_confirm", "text": text}
            if self.is_negative(user_input):
                self.collected_data.selected_appointment = None
                self.collected_data.appointment_confirmed = False
                self.current_step = self.get_next_step()
                display_text, mapping = self.format_appointments()
                text = f"No problem. Here are some other appointment options.\n{display_text}"
                self.add_to_history("assistant", text)
                return {"type": "show_appointments", "text": text, "max": len(mapping)}
            text = "Please reply yes to confirm this appointment or no to see other options."
            self.add_to_history("assistant", text)
            return {"type": "reply", "text": text}

        # Extract structured fields from user message
        extracted = self.extract(user_input)
        # Save extracted info
        extraction_error = self.apply_extracted(extracted)
        if extraction_error and self.current_step == "date_of_birth":
            self.collected_data.patient.date_of_birth = None
            text = f"That date of birth doesn't look valid. {extraction_error}"
            self.add_to_history("assistant", text)
            return {"type": "validation_error", "text": text}
    
        address_parts = []
        a = self.collected_data.address
        if a.street:
            address_parts.append(a.street)
        if a.city:
            address_parts.append(a.city)
        if a.state:
            address_parts.append(a.state)
        if a.zip_code:
            address_parts.append(a.zip_code)
        
        if address_parts and not a.formatted_address:
            full_address = ", ". join(address_parts)
            is_valid, formatted_address, components, error_detail = self.validate_address(a)
            if is_valid:
                self.collected_data.address.formatted_address = formatted_address
                if components:
                    street_number = components.get("street_number")
                    route = components.get("route")
                    if street_number and route:
                        self.collected_data.address.street = f"{street_number} {route}"
                    self.collected_data.address.city = components.get("city", self.collected_data.address.city)
                    self.collected_data.address.state = components.get("state", self.collected_data.address.state)
                    self.collected_data.address.zip_code = components.get("zip_code", self.collected_data.address.zip_code)
                self.current_step = self.get_next_step()
                if self.current_step == "appointment_selection":
                    display_text, mapping = self.format_appointments()
                    text = f"Thanks, I validated your address as {formatted_address}.\n{display_text}"
                    self.add_to_history("assistant", text)
                    return {"type": "show_appointments", "text": text, "max": len(mapping)}
                text = f"Thanks, I validated your address as {formatted_address}."
                self.add_to_history("assistant", text)
                return {"type": "reply", "text": text}
            else:
                self.current_step = "address"
                text = (
                    "I'm having trouble validating that address. Please provide your complete "
                    "street, city, state, and zip code."
                )
                if error_detail:
                    text += f" Validation detail: {error_detail}."
                self.add_to_history("assistant", text)
                return {"type": "validation_error", "text": text}

        self.current_step = self.get_next_step()

        if self.current_step == "appointment_selection" and not self.collected_data.selected_appointment:
            numbers = re.findall(r"\b(\d+)\b", user_input)
            if numbers:
                x = numbers[0]
                apt = self.get_appointment_by_number(int(x))
                if apt:
                    self.collected_data.selected_appointment = apt
                    self.current_step = self.get_next_step()
                    text = f"Great, I've reserved {apt.provider_name} on {apt.datetime.strftime('%A, %B %d at %I:%M %p')} for you. Does that work?"
                    self.add_to_history("assistant", text)
                    return {"type": "reply", "text": text}
                text = "I couldn't match that appointment number. Please choose one of the listed options."
                self.add_to_history("assistant", text)
                display_text, mapping = self.format_appointments()
                return {"type": "show_appointments", "text": f"{text}\n{display_text}", "max": len(mapping)}
            display_text, mapping = self.format_appointments()
            return {"type": "show_appointments", "text": display_text, "max": len(mapping)}

        response = self.prompt_for_step(self.current_step)
        self.add_to_history("assistant", response)
        return {"type": "reply", "text": response}

def main():
    print("     APPOINTMENT SCHEDULING ASSISTANT     ")
    print("----------------------------------------")
    print("Enter 'exit','quit', or 'q' to end the conversation")
    print("----------------------------------------")

    openai_api_key = os.getenv("OPENAI_API_KEY")
    google_maps_api_key = os.getenv("GOOGLE_MAPS_API_KEY")

    # prepare appointments
    appointments = generate_mock_appointments()
    # initialize agent
    agent = AppointmentAgent(
        openai_api_key=openai_api_key,
        appointments=appointments,
        google_maps_api_key=google_maps_api_key
    )
    # start conversation
    greeting = "Hi there! I'm here to help you schedule your medical appointment. To get started, could you please tell me your full name?"
    print(f"Assistant: {greeting}")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ["exit", "quit", "q"]:
            print("\nAssistant: Thanks for your time. Feel free to come back when you're ready to schedule!")
            break
        event = agent.chat(user_input)
        etype = event.get("type")
        if etype == "reply" or etype == "reply_and_ready_to_confirm":
            print(f"\nAssistant: {event.get('text')}\n")
            if etype == "reply_and_ready_to_confirm":
                print(agent.display_confirmation())
                break
        elif etype == "validation_error":
            print(f"\nAssistant: {event.get('text')}\n")
        elif etype == "show_appointments":
            print(event.get('text'))
            print(f"\nAssistant: Which appointment would you prefer? Just tell me the number.\n")
        else:
            print(f"\nAssistant: {event.get('text')}\n")

if __name__ == "__main__":
    main()
