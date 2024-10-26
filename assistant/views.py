# assistant/views.py
from django.http import JsonResponse
from django.shortcuts import render
from .nlp_helper import generate_response
from .voice_assistant import text_to_speech


def voice_command_view(request):
    if request.method == "POST":
        # Step 1: Get the user input from POST data
        user_input = request.POST.get("user_input", "")

        # Step 2: Generate AI response using NLP
        response_text = generate_response(user_input)

        # Step 3: Optionally, convert the response to speech and play it (if desired)
        text_to_speech(response_text)

        # Step 4: Return the AI response as JSON
        return JsonResponse({"user_input": user_input, "response": response_text})

    return JsonResponse({"error": "Invalid request method"}, status=400)


def home(request):
    return render(request, "assistant/home.html")
