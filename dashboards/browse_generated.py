import json

import pandas as pd
import fortepyan as ff
import streamlit as st
import streamlit_pianoroll

import data.database_manager as dm


def main():
    st.title("MIDI Transformers Database Browser")

    tab1, tab2, tab3, tab4 = st.tabs(["Model Predictions", "Models", "Generation Parameters", "Prompt Notes"])

    with tab1:
        st.header("Model Predictions")

        models_df = dm.get_all_models()
        model_names = models_df["name"].tolist()

        selected_model_name = st.selectbox("Select Model", model_names, key="model")

        if selected_model_name:
            selected_model = models_df[models_df["name"] == selected_model_name].iloc[0]
            if pd.notna(selected_model["wandb_link"]):
                st.link_button("View Model on W&B", url=selected_model["wandb_link"])
            else:
                st.write("No W&B link available for this model")

            selected_model_id = selected_model["id"]

            # Fetch prompts for the selected model
            prompts = dm.get_prompts_for_model(model_id=selected_model_id)
            selected_prompt_id = st.selectbox("Select Prompt", prompts["id"].tolist())

            if selected_prompt_id:
                full_prompt = dm.get_prompt(prompt_id=selected_prompt_id)
                st.write(full_prompt)

                if st.button("Get Predictions"):
                    # Fetch all predictions for the selected model and prompt
                    predictions_df = dm.get_model_predictions(
                        model_filters={"id": selected_model_id}, prompt_filters={"id": selected_prompt_id}
                    )

                    if not predictions_df.empty:
                        for _, row in predictions_df.iterrows():
                            parameters = dm.get_parameters(row["parameters_id"]).to_dict(orient="records")
                            st.json(parameters, expanded=False)
                            notes = json.loads(row["generated_notes"])
                            notes_df = pd.DataFrame(notes)
                            piece = ff.MidiPiece(df=notes_df)

                            streamlit_pianoroll.from_fortepyan(piece=piece)
                            st.divider()  # Add a divider between predictions
                    else:
                        st.write("No predictions found for this prompt and model combination.")

    # The rest of the tabs remain unchanged
    with tab2:
        st.header("Models")
        models_df = dm.get_all_models()
        st.write(models_df)

        st.subheader("Purge Model")
        model_to_purge = st.selectbox("Select a model to purge", models_df["name"].tolist())
        if st.button("Purge Selected Model"):
            if st.checkbox("Are you sure? This action cannot be undone."):
                try:
                    dm.purge_model(model_to_purge)
                    st.success(f"Model '{model_to_purge}' has been purged successfully.")
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
