import streamlit as st
import importlib
import sys
from pathlib import Path

# Add current directory to Python path to enable dynamic imports
sys.path.append(str(Path(__file__).parent))


def clear_sidebar():
    # Clears the sidebar and resets session variables
    # Used when switching between assignments or questions
    for key in list(st.session_state.keys()):
        if key.startswith("analysis_") or "select" in key:
            del st.session_state[key]
    st.sidebar.empty()


def load_module(module_name):
    # Loads a module and runs its main() function
    # module_name: name of module to import and run
    try:
        module = importlib.import_module(module_name)
        if hasattr(module, "main"):
            module.main()  # Run the main function
        else:
            st.warning(f"The module '{module_name}' doesn't have a main() function.")
    except Exception as e:
        st.error(f"Error whilst loading module '{module_name}': {str(e)}")


def main():
    # Main app function - handles assignment switching and content updates
    
    # Set up page layout
    st.set_page_config(
        page_title="ARI5122 - Financial Engineering",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Show main title
    st.markdown("## ARI5122 - Financial Engineering")
    
    # Set up tracking variables if they don't exist
    if "active_assignment" not in st.session_state:
        st.session_state.active_assignment = "Assignment 1"
    if "active_question" not in st.session_state:
        st.session_state.active_question = None

    # Create assignment selection in sidebar
    st.sidebar.markdown("### Select an Assignment")
    selected_assignment = st.sidebar.radio(
        "Assignments",
        options=["Assignment 1", "Assignment 2"],
        index=0 if st.session_state.active_assignment == "Assignment 1" else 1,
        key="assignment_radio"
    )

    # Reset when changing assignments
    if st.session_state.active_assignment != selected_assignment:
        st.session_state.active_assignment = selected_assignment
        st.session_state.active_question = None
        clear_sidebar()  # Clear for new assignment

    # Set up questions for each assignment
    if selected_assignment == "Assignment 1":
        st.markdown("### Assignment 1")
        questions = {
            "Select a question": None,
            "Question 1": "Ass1A",
            "Question 2": "Ass1B"
        }
    elif selected_assignment == "Assignment 2":
        st.markdown("### Assignment 2")
        questions = {
            "Select a question": None,
            "Question 1": "Ass2A",
            "Question 2": "Ass2B",
            "Question 3": "Ass2C",
            "Question 4": "Ass2D"
        }

    # Create question dropdown in sidebar
    selected_question = st.selectbox(
        f"Choose a question for {selected_assignment}:",
        options=list(questions.keys()),
        key="question_select"
    )

    # Load selected question's module
    if selected_question != "Select a question":
        if st.session_state.active_question != selected_question:
            st.session_state.active_question = selected_question
            clear_sidebar()  # Clear when new question selected
        
        module_name = questions[selected_question]
        load_module(module_name)


if __name__ == "__main__":
    main()