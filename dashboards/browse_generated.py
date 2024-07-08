import json

import pandas as pd
import fortepyan as ff
import streamlit as st
import streamlit_pianoroll

import dashboards.common.database_manager as dm


def main():
    # Streamlit App Title
    st.title("MIDI Transformers Database Browser")

    # Create tabs for navigation
    tab1, tab2, tab3, tab4 = st.tabs(["Model Predictions", "Models", "Generation Parameters", "Prompt Notes"])

    with tab1:
        st.header("Model Predictions")

        models_df = dm.get_all_models()
        model_names = models_df["name"].tolist()

        col1, col2 = st.columns(2)

        with col1:
            selected_model_name_1 = st.selectbox(
                "Select Model 1",
                model_names,
                key="model_1",
            )
            if selected_model_name_1:
                selected_model_1 = models_df[models_df["name"] == selected_model_name_1].iloc[0]
                if pd.notna(selected_model_1["wandb_link"]):
                    st.link_button("View Model 1 on W&B", url=selected_model_1["wandb_link"])
                else:
                    st.write("No W&B link available for Model 1")

        with col2:
            selected_model_name_2 = st.selectbox(
                "Select Model 2",
                model_names,
                key="model_2",
            )
            if selected_model_name_2:
                selected_model_2 = models_df[models_df["name"] == selected_model_name_2].iloc[0]
                if pd.notna(selected_model_2["wandb_link"]):
                    st.link_button("View Model 2 on W&B", url=selected_model_2["wandb_link"])
                else:
                    st.write("No W&B link available for Model 2")

        if selected_model_name_1 and selected_model_name_2:
            # Get the selected model_ids
            selected_model_id_1 = models_df[models_df["name"] == selected_model_name_1].iloc[0]["id"]
            selected_model_id_2 = models_df[models_df["name"] == selected_model_name_2].iloc[0]["id"]

            # Fetch common prompt_ids and parameters_ids for the selected models
            prompt_ids, parameters_ids = dm.get_common_prompts_and_parameters_for_models(
                model_id_1=selected_model_id_1,
                model_id_2=selected_model_id_2,
            )

            selected_prompt_id = st.selectbox("Select Prompt ID", prompt_ids)
            full_prompt = dm.get_prompt(prompt_id=selected_prompt_id)
            st.write(full_prompt)
            selected_parameters_id = st.selectbox("Select Parameters ID", parameters_ids)
            full_parameters = dm.get_parameters(parameters_id=selected_parameters_id)
            st.write(full_parameters)

            filters_1 = {}
            filters_2 = {}

            if selected_model_name_1:
                filters_1["model_filters"] = {"name": selected_model_name_1}
            if selected_prompt_id:
                filters_1["prompt_filters"] = {"id": selected_prompt_id}
            if selected_parameters_id:
                filters_1["parameter_filters"] = {"id": selected_parameters_id}

            if selected_model_name_2:
                filters_2["model_filters"] = {"name": selected_model_name_2}
            if selected_prompt_id:
                filters_2["prompt_filters"] = {"id": selected_prompt_id}
            if selected_parameters_id:
                filters_2["parameter_filters"] = {"id": selected_parameters_id}

            if st.button("Get Predictions"):
                predictions_df_1 = dm.get_model_predictions(
                    model_filters=filters_1.get("model_filters"),
                    prompt_filters=filters_1.get("prompt_filters"),
                    parameter_filters=filters_1.get("parameter_filters"),
                )

                predictions_df_2 = dm.get_model_predictions(
                    model_filters=filters_2.get("model_filters"),
                    prompt_filters=filters_2.get("prompt_filters"),
                    parameter_filters=filters_2.get("parameter_filters"),
                )

                notes_1 = json.loads(predictions_df_1["generated_notes"][0])
                notes_2 = json.loads(predictions_df_2["generated_notes"][0])

                notes_1 = pd.DataFrame(notes_1)
                notes_2 = pd.DataFrame(notes_2)

                piece_1 = ff.MidiPiece(df=notes_1)
                piece_2 = ff.MidiPiece(df=notes_2)

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader(f"Predictions for {selected_model_name_1}")
                    streamlit_pianoroll.from_fortepyan(piece=piece_1)

                with col2:
                    st.subheader(f"Predictions for {selected_model_name_2}")
                    streamlit_pianoroll.from_fortepyan(piece=piece_2)

            # Display the selected values
            st.write(f"Selected Model 1: {selected_model_name_1}")
            st.write(f"Selected Model 2: {selected_model_name_2}")
            st.write(f"Selected Prompt ID: {selected_prompt_id}")
            st.write(f"Selected Parameters ID: {selected_parameters_id}")
        else:
            st.write("Please select both models to see common prompts and parameters.")

    with tab2:
        st.header("Models")
        models_df = dm.get_all_models()
        st.write(models_df)

        # Add a section for purging a model
        st.subheader("Purge Model")
        model_to_purge = st.selectbox("Select a model to purge", models_df["name"].tolist())
        if st.button("Purge Selected Model"):
            if st.checkbox("Are you sure? This action cannot be undone."):
                try:
                    dm.purge_model(model_to_purge)
                    st.success(f"Model '{model_to_purge}' has been purged successfully.")
                    # Refresh the models dataframe
                    models_df = dm.get_all_models()
                    st.write(models_df)
                except Exception as e:
                    st.error(f"An error occurred while purging the model: {str(e)}")
            else:
                st.warning("Please confirm the action by checking the box.")

    with tab3:
        st.header("Generation Parameters")
        parameters_df = dm.get_all_generation_parameters()
        st.write(parameters_df)

    with tab4:
        st.header("Prompt Notes")
        prompts_df = dm.get_all_prompt_notes()
        st.write(prompts_df)


if __name__ == "__main__":
    main()
