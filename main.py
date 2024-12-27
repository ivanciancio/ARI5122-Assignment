import streamlit as st
import importlib
import sys
from pathlib import Path

# Add the current directory to the Python path for dynamic imports
sys.path.append(str(Path(__file__).parent))


def clear_sidebar():
    """
    Clears the sidebar content and resets relevant session state variables.
    This ensures that the sidebar is updated dynamically when switching
    between assignments or questions.
    """
    for key in list(st.session_state.keys()):
        if key.startswith("analysis_") or "select" in key:
            del st.session_state[key]
    st.sidebar.empty()


def load_module(module_name):
    """
    Dynamically imports a module and executes its main() function.
    
    Parameters:
        module_name (str): The name of the module to import and execute.
    """
    try:
        module = importlib.import_module(module_name)
        if hasattr(module, "main"):
            module.main()  # Execute the main function of the module
        else:
            st.warning(f"The module '{module_name}' does not have a main() function.")
    except Exception as e:
        st.error(f"Error while loading module '{module_name}': {str(e)}")


def main():
    """
    Main function to render the Streamlit app. It allows users to switch
    between assignments and questions dynamically, with content updating
    appropriately.
    """
    # Configure the page layout and title
    st.set_page_config(
        page_title="ARI5122 - Financial Engineering",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Display the main title of the application
    st.markdown("## ARI5122 - Financial Engineering")
    
    # Initialise session state variables for assignment and question tracking
    if "active_assignment" not in st.session_state:
        st.session_state.active_assignment = "Assignment 1"
    if "active_question" not in st.session_state:
        st.session_state.active_question = None

    # Sidebar: Assignment selection using radio buttons
    st.sidebar.markdown("### Select an Assignment")
    selected_assignment = st.sidebar.radio(
        "Assignments",
        options=["Assignment 1", "Assignment 2"],
        index=0 if st.session_state.active_assignment == "Assignment 1" else 1,
        key="assignment_radio"
    )

    # Reset the state when switching between assignments
    if st.session_state.active_assignment != selected_assignment:
        st.session_state.active_assignment = selected_assignment
        st.session_state.active_question = None
        clear_sidebar()  # Clear sidebar content for the new assignment

    # Define questions for each assignment
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

    # Sidebar: Question selection using a dropdown menu
    selected_question = st.selectbox(
        f"Choose a question for {selected_assignment}:",
        options=list(questions.keys()),
        key="question_select"
    )

    # Load the module corresponding to the selected question
    if selected_question != "Select a question":
        if st.session_state.active_question != selected_question:
            st.session_state.active_question = selected_question
            clear_sidebar()  # Clear sidebar when a new question is selected
        
        module_name = questions[selected_question]
        load_module(module_name)


if __name__ == "__main__":
    main()