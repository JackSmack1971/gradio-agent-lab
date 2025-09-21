#!/usr/bin/env python3
"""
Comprehensive validation script for Gradio Agent Lab implementation.
Tests the complete reproduction against paper specifications.
"""

import os
import sys
import json
import inspect
from unittest.mock import Mock, patch
import gradio as gr

# Mock the OpenRouter API key for testing
os.environ['OPENROUTER_API_KEY'] = 'test-key-for-validation'

# Import the agent_lab module to access the demo
import agent_lab

def test_ui_component_structure():
    """Test that all required UI components are present and correctly configured."""
    print("✓ Testing UI component structure...")

    demo = agent_lab.demo

    # Check that it's a Blocks instance
    assert isinstance(demo, gr.Blocks), "Demo should be a Gradio Blocks instance"

    # Get all components from the demo
    components = []
    def collect_components(block):
        components.append(block)
        if hasattr(block, 'children'):
            for child in block.children:
                collect_components(child)

    collect_components(demo)

    # Required component types
    required_types = [
        gr.Markdown,  # Title markdown
        gr.Sidebar,   # Model metadata sidebar
        gr.Dropdown,  # Model selection
        gr.Slider,    # Temperature and top_p sliders
        gr.Number,    # Max tokens and seed
        gr.Textbox,   # System prompt and sidebar model ID
        gr.ChatInterface,  # Main chat interface
        gr.Button     # Refresh button
    ]

    found_types = [type(comp) for comp in components]

    for req_type in required_types:
        assert req_type in found_types, f"Missing required component type: {req_type.__name__}"

    print("✓ All required UI components present")
    return True

def test_sidebar_configuration():
    """Test sidebar configuration matches paper specifications."""
    print("✓ Testing sidebar configuration...")

    demo = agent_lab.demo

    def find_sidebar(block):
        if isinstance(block, gr.Sidebar):
            return block
        if hasattr(block, 'children'):
            for child in block.children:
                result = find_sidebar(child)
                if result:
                    return result
        return None

    sidebar = find_sidebar(demo)
    assert sidebar is not None, "Sidebar not found"

    # Check sidebar properties
    assert sidebar.label == "Model Metadata", f"Wrong sidebar label: {sidebar.label}"
    assert sidebar.open == True, "Sidebar should be open by default"
    assert sidebar.width == 320, f"Wrong sidebar width: {sidebar.width}"

    print("✓ Sidebar configuration correct")
    return True

def test_chat_interface_configuration():
    """Test ChatInterface configuration matches paper specifications."""
    print("✓ Testing ChatInterface configuration...")

    demo = agent_lab.demo

    def find_chat_interface(block):
        if isinstance(block, gr.ChatInterface):
            return block
        if hasattr(block, 'children'):
            for child in block.children:
                result = find_chat_interface(child)
                if result:
                    return result
        return None

    chat = find_chat_interface(demo)
    assert chat is not None, "ChatInterface not found"

    # Check ChatInterface properties from paper
    assert chat.type == "messages", f"Wrong chat type: {chat.type}"
    assert chat.fill_height == True, "ChatInterface should fill height"
    assert chat.autofocus == True, "ChatInterface should have autofocus"

    # Check button configuration
    assert chat.retry_btn is None, "Retry button should be disabled"
    assert chat.undo_btn == "Delete last", f"Wrong undo button text: {chat.undo_btn}"
    assert chat.submit_btn == "Send", f"Wrong submit button text: {chat.submit_btn}"
    assert chat.stop_btn == "Stop", f"Wrong stop button text: {chat.stop_btn}"

    # Check additional inputs
    assert len(chat.additional_inputs) == 6, f"Wrong number of additional inputs: {len(chat.additional_inputs)}"

    print("✓ ChatInterface configuration correct")
    return True

def test_parameter_controls():
    """Test parameter control configurations."""
    print("✓ Testing parameter controls...")

    demo = agent_lab.demo

    def find_sliders_and_numbers(block, controls=None):
        if controls is None:
            controls = []
        if isinstance(block, (gr.Slider, gr.Number)):
            controls.append(block)
        if hasattr(block, 'children'):
            for child in block.children:
                find_sliders_and_numbers(child, controls)
        return controls

    controls = find_sliders_and_numbers(demo)

    # Should have 4 controls: 2 sliders (temperature, top_p) and 2 numbers (max_tokens, seed)
    assert len(controls) == 4, f"Wrong number of parameter controls: {len(controls)}"

    # Check sliders
    sliders = [c for c in controls if isinstance(c, gr.Slider)]
    assert len(sliders) == 2, f"Wrong number of sliders: {len(sliders)}"

    # Temperature slider
    temp_slider = next((s for s in sliders if s.label == "temperature"), None)
    assert temp_slider is not None, "Temperature slider not found"
    assert temp_slider.minimum == 0.0, f"Wrong temperature min: {temp_slider.minimum}"
    assert temp_slider.maximum == 2.0, f"Wrong temperature max: {temp_slider.maximum}"
    assert temp_slider.value == 0.7, f"Wrong temperature default: {temp_slider.value}"
    assert temp_slider.step == 0.05, f"Wrong temperature step: {temp_slider.step}"

    # Top-p slider
    topp_slider = next((s for s in sliders if s.label == "top_p"), None)
    assert topp_slider is not None, "Top-p slider not found"
    assert topp_slider.minimum == 0.0, f"Wrong top_p min: {topp_slider.minimum}"
    assert topp_slider.maximum == 1.0, f"Wrong top_p max: {topp_slider.maximum}"
    assert topp_slider.value == 1.0, f"Wrong top_p default: {topp_slider.value}"
    assert topp_slider.step == 0.01, f"Wrong top_p step: {topp_slider.step}"

    # Check numbers
    numbers = [c for c in controls if isinstance(c, gr.Number)]
    assert len(numbers) == 2, f"Wrong number of number inputs: {len(numbers)}"

    # Max tokens
    maxtokens_num = next((n for n in numbers if "max_tokens" in n.label), None)
    assert maxtokens_num is not None, "Max tokens number input not found"
    assert maxtokens_num.value == 1024, f"Wrong max_tokens default: {maxtokens_num.value}"

    # Seed
    seed_num = next((n for n in numbers if "seed" in n.label), None)
    assert seed_num is not None, "Seed number input not found"
    assert seed_num.value == 0, f"Wrong seed default: {seed_num.value}"

    print("✓ Parameter controls configured correctly")
    return True

def test_model_dropdown():
    """Test model dropdown configuration."""
    print("✓ Testing model dropdown...")

    demo = agent_lab.demo

    def find_dropdown(block):
        if isinstance(block, gr.Dropdown):
            return block
        if hasattr(block, 'children'):
            for child in block.children:
                result = find_dropdown(child)
                if result:
                    return result
        return None

    dropdown = find_dropdown(demo)
    assert dropdown is not None, "Model dropdown not found"

    assert dropdown.label == "Model (OpenRouter)", f"Wrong dropdown label: {dropdown.label}"
    assert len(dropdown.choices) > 0, "Dropdown should have choices"

    # Check that preferred models are prioritized
    choices_values = [choice[1] for choice in dropdown.choices]
    preferred = ["openrouter/auto", "anthropic/claude-3.5-sonnet", "openai/gpt-4o-mini"]
    for pref in preferred:
        assert pref in choices_values, f"Preferred model {pref} not in choices"

    print("✓ Model dropdown configured correctly")
    return True

def test_system_prompt_textbox():
    """Test system prompt textbox configuration."""
    print("✓ Testing system prompt textbox...")

    demo = agent_lab.demo

    def find_textboxes(block, textboxes=None):
        if textboxes is None:
            textboxes = []
        if isinstance(block, gr.Textbox):
            textboxes.append(block)
        if hasattr(block, 'children'):
            for child in block.children:
                find_textboxes(child, textboxes)
        return textboxes

    textboxes = find_textboxes(demo)

    # Should have at least 2 textboxes: system prompt and sidebar model ID
    assert len(textboxes) >= 2, f"Not enough textboxes: {len(textboxes)}"

    # Find system prompt textbox
    system_prompt = None
    for tb in textboxes:
        if tb.label == "System Instructions":
            system_prompt = tb
            break

    assert system_prompt is not None, "System prompt textbox not found"
    assert system_prompt.lines == 5, f"Wrong number of lines: {system_prompt.lines}"
    assert "code-review copilot" in system_prompt.placeholder, "Wrong placeholder text"

    print("✓ System prompt textbox configured correctly")
    return True

def test_sidebar_functionality():
    """Test sidebar update functionality."""
    print("✓ Testing sidebar functionality...")

    # Mock model data
    mock_model = {
        "id": "test-model",
        "name": "Test Model",
        "owned_by": "test-provider",
        "context_length": 1000,
        "pricing": {"prompt": 1.0, "completion": 2.0},
        "description": "A test model"
    }

    with patch('agent_lab.find_model', return_value=mock_model):
        # Test update_sidebar function
        result = agent_lab.update_sidebar("test-model")

        assert len(result) == 2, "update_sidebar should return 2 values"
        model_id, markdown = result

        assert model_id == "test-model", f"Wrong model ID: {model_id}"
        assert "### Model" in markdown, "Markdown should contain model header"
        assert "Test Model" in markdown, "Markdown should contain model name"
        assert "test-provider" in markdown, "Markdown should contain provider"

    print("✓ Sidebar functionality works correctly")
    return True

def test_refresh_functionality():
    """Test model refresh functionality."""
    print("✓ Testing refresh functionality...")

    # Mock new model data
    mock_response = Mock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        "data": [
            {"id": "new-model", "name": "New Model"},
            {"id": "openrouter/auto", "name": "Auto"}
        ]
    }

    with patch('requests.get', return_value=mock_response):
        # Clear cache and test refresh
        agent_lab.fetch_models_raw.cache_clear()
        result = agent_lab.refresh_models("old-model")

        assert len(result) == 3, "refresh_models should return 3 values"
        new_dropdown, new_model_id, new_markdown = result

        # Check that dropdown choices were updated
        assert len(new_dropdown['choices']) > 0, "New dropdown should have choices"
        assert new_model_id == "openrouter/auto", "Should fallback to first choice when current model not found"

    print("✓ Refresh functionality works correctly")
    return True

def test_paper_compliance():
    """Test compliance with paper specifications."""
    print("✓ Testing paper compliance...")

    # Check that all major features from the paper are implemented

    # 1. Gradio ChatInterface with streaming
    assert hasattr(agent_lab, 'reply_fn'), "reply_fn function should exist"
    assert hasattr(agent_lab, 'stream_openrouter'), "stream_openrouter function should exist"

    # 2. Model selection and metadata
    assert hasattr(agent_lab, 'list_model_choices'), "list_model_choices function should exist"
    assert hasattr(agent_lab, 'build_model_markdown'), "build_model_markdown function should exist"

    # 3. Parameter controls
    assert hasattr(agent_lab, 'MODEL_CHOICES'), "MODEL_CHOICES should be defined"

    # 4. UI Layout matches paper description
    demo = agent_lab.demo
    assert demo.title == "Agent Lab · OpenRouter", f"Wrong title: {demo.title}"
    assert demo.theme == "soft", f"Wrong theme: {demo.theme}"

    print("✓ Implementation complies with paper specifications")
    return True

def run_comprehensive_validation():
    """Run all comprehensive validation tests."""
    print("=" * 70)
    print("COMPREHENSIVE GRADIO AGENT LAB VALIDATION")
    print("=" * 70)

    tests = [
        test_ui_component_structure,
        test_sidebar_configuration,
        test_chat_interface_configuration,
        test_parameter_controls,
        test_model_dropdown,
        test_system_prompt_textbox,
        test_sidebar_functionality,
        test_refresh_functionality,
        test_paper_compliance,
    ]

    results = []
    for test in tests:
        try:
            print(f"\n{test.__name__.replace('_', ' ').title()}:")
            result = test()
            results.append(result)
            print(f"✓ PASSED")
        except Exception as e:
            print(f"✗ FAILED: {e}")
            results.append(False)

    print("\n" + "=" * 70)
    print("COMPREHENSIVE VALIDATION RESULTS")
    print("=" * 70)

    passed = sum(results)
    total = len(results)
    accuracy = (passed / total) * 100

    print(f"Tests passed: {passed}/{total} ({accuracy:.1f}%)")

    if accuracy >= 85:
        print("✓ COMPREHENSIVE VALIDATION SUCCESS: Meets quality threshold")
        status = "SUCCESS"
    else:
        print("✗ COMPREHENSIVE VALIDATION ISSUES: Below quality threshold")
        status = "ISSUES"

    # Detailed breakdown
    test_names = [
        "UI Component Structure",
        "Sidebar Configuration",
        "ChatInterface Configuration",
        "Parameter Controls",
        "Model Dropdown",
        "System Prompt Textbox",
        "Sidebar Functionality",
        "Refresh Functionality",
        "Paper Compliance"
    ]

    print("\nDetailed Results:")
    for name, result in zip(test_names, results):
        status_icon = "✓" if result else "✗"
        print(f"{status_icon} {name}")

    print(f"\nOverall Status: {status}")
    print(".1f")

    return accuracy >= 85

if __name__ == "__main__":
    success = run_comprehensive_validation()
    sys.exit(0 if success else 1)