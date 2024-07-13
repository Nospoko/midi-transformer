import json

import pandas as pd
import fortepyan as ff
import streamlit as st
import streamlit_pianoroll

import data.database_manager as database_manager


def main():
    st.title("MIDI Transformers Database Browser")

    tab1, tab2, tab3, tab4 = st.tabs(["Model Predictions", "Models", "Generation Parameters", "Prompt Notes"])

    with tab1:
        st.header("Model Predictions")

        models_df = database_manager.get_all_models()
        model_names = models_df["name"].tolist()

        selected_model_name = st.selectbox("Select Model", model_names, key="model")

        if selected_model_name:
            selected_model = models_df[models_df["name"] == selected_model_name].iloc[0]
            if pd.notna(selected_model["wandb_link"]):
                st.link_button("View Model on W&B", url=selected_model["wandb_link"])
            else:
                st.write("No W&B link available for this model")

            selected_model_id = selected_model["model_id"]

            # Fetch prompts for the selected model
            prompts = database_manager.get_prompts_for_model(model_id=selected_model_id)
            selected_prompt_id = st.selectbox("Select Prompt", prompts["prompt_id"].tolist())

            if selected_prompt_id:
                full_prompt = database_manager.get_prompt(prompt_id=selected_prompt_id)
                st.write(full_prompt)

                # Fetch all predictions for the selected model and prompt
                predictions_df = database_manager.get_model_predictions(
                    model_filters={"model_id": selected_model_id}, prompt_filters={"prompt_id": selected_prompt_id}
                )

                if not predictions_df.empty:
                    for _, row in predictions_df.iterrows():
                        parameters = database_manager.get_parameters(row["parameters_id"]).iloc[0].to_dict()
                        prompt = database_manager.get_prompt(row["prompt_id"]).iloc[0]

                        st.json(parameters, expanded=False)
                        prompt_notes = json.loads(prompt["prompt_notes"])
                        prompt_notes_df = pd.DataFrame(prompt_notes)

                        bass_notes = json.loads(row["generated_notes"])
                        bass_notes_df = pd.DataFrame(bass_notes)
                        bass_piece = ff.MidiPiece(df=bass_notes_df)

                        prompt_piece = ff.MidiPiece(df=prompt_notes_df)
                        streamlit_pianoroll.from_fortepyan(piece=prompt_piece, secondary_piece=bass_piece)
                        st.divider()  # Add a divider between predictions
                else:
                    st.write("No predictions found for this prompt and model combination.")

    # The rest of the tabs remain unchanged
    with tab2:
        st.header("Models")
        models_df = database_manager.get_all_models()
        st.write(models_df)

        st.subheader("Purge Model")
        model_to_purge = st.selectbox("Select a model to purge", models_df["name"].tolist())
        if st.button("Purge Selected Model"):
            if st.checkbox("Are you sure? This action cannot be undone."):
                try:
                    database_manager.purge_model(model_to_purge)
                    st.success(f"Model '{model_to_purge}' has been purged successfully.")
                    models_df = database_manager.get_all_models()
                    st.write(models_df)
                except Exception as e:
                    st.error(f"An error occurred while purging the model: {str(e)}")
            else:
                st.warning("Please confirm the action by checking the box.")

    with tab3:
        st.header("Generation Parameters")
        parameters_df = database_manager.get_all_generation_parameters()
        st.write(parameters_df)

    with tab4:
        st.header("Prompt Notes")
        prompts_df = database_manager.get_all_prompt_notes()
        st.write(prompts_df)


if __name__ == "__main__":
    main()
