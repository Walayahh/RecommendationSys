### IMPORTS ###
from flask import Flask, render_template, request, send_from_directory, jsonify
import logging
import os
import pandas as pd
from rowwise_content_recommender import (
    load_and_prepare_perfumes,
    build_tfidf_matrix,
    get_content_based_recommendations
)
from rowwise_collab_recommender import (
    build_item_user_matrix,
    get_collab_recommendations
)
from weighted_rating import get_top_rated_recommendations
from popularity_rating import get_popular_perfume_recommendations

### END IMPORTS ###

app = Flask(__name__)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def to_display_list(recs_list):
        result = []
        for perfume_name in recs_list:
            # Lookup using the "Perfumes" column from the CSV
            row = perfumes_df.loc[perfumes_df['Perfumes'] == perfume_name]
            if not row.empty:
                # "Perfume" is used for the display name.
                name = row.iloc[0]['Perfume']
                brand = row.iloc[0]['Brand']
                mainaccord1 = row.iloc[0].get('mainaccord1', '')
                mainaccord2 = row.iloc[0].get('mainaccord2', '')
                mainaccord3 = row.iloc[0].get('mainaccord3', '')
                mainaccord4 = row.iloc[0].get('mainaccord4', '')
                mainaccord5 = row.iloc[0].get('mainaccord5', '')
            else:
                name = perfume_name
                brand = "Unknown"
                mainaccord1 = mainaccord2 = mainaccord3 = mainaccord4 = mainaccord5 = ""
            result.append({
                "name": name,
                "brand": brand,
                "mainaccord1": mainaccord1,
                "mainaccord2": mainaccord2,
                "mainaccord3": mainaccord3,
                "mainaccord4": mainaccord4,
                "mainaccord5": mainaccord5,
            })
        return result
    
# Load data and build recommendation structures.
logger.info("Loading data for content-based recommender...")
perfumes_df = load_and_prepare_perfumes("perfumes_updated.csv", delimiter=";", encoding="latin-1")
perfume_counts_df = pd.read_csv("perfume_counts.csv")
tfidf_matrix, index_to_name, name_to_index = build_tfidf_matrix(perfumes_df)

logger.info("Loading data for item-based collaborative recommender...")
item_user_df, item_names, user_names, item_to_index = build_item_user_matrix(
    users_csv="users.csv",
    perfumes_csv="perfumes_updated.csv",
    user_col="UserID",
    delimiter_users=",",
    delimiter_perfumes=";",
    encoding_users="latin-1",
    encoding_perfumes="latin-1"
)


@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

@app.route('/', methods=['GET'])
def index():
    """
    Render the main index page where users can pick perfumes and submit for recommendations.
    """
    return render_template('index.html')

@app.route('/recommendations', methods=['POST'])
def recommendations():
    """
    1. Receive user-selected perfumes and gender.
    2. Generate both content-based and collaborative recommendations.
    3. Convert raw recommendation names (looked up via the CSV) into display dictionaries.
    4. Render recommendations.html with the processed data.
    """
    perfumes_str = request.form.get('perfumes', '')
    gender = request.form.get('gender', '')
    print('gender is gender')
    user_selected = [p.strip() for p in perfumes_str.split(',') if p.strip()]

    logger.info("User selected perfumes: %s, gender=%s", user_selected, gender)

    # Content-based recommendations.
    content_based_recs = get_content_based_recommendations(
        selected_perfumes=user_selected,
        tfidf_matrix=tfidf_matrix,
        name_to_index=name_to_index,
        index_to_name=index_to_name,
        top_k_each=200,
        final_n=10
    )
    logger.info("Content-based recommendations: %s", content_based_recs)

    # Collaborative recommendations.
    collab_recs = get_collab_recommendations(
        selected_items=user_selected,
        item_user_df=item_user_df,
        item_to_index=item_to_index,
        top_k_each=200,
        final_n=10
    )
    
    logger.info("Collaborative recommendations: %s", collab_recs)

    top_k = int(request.args.get('top_k', 10))
    
    logger.info(f"Getting most popular perfumes for gender={gender}, top_k={top_k}")
    
    popular_perfumes = get_popular_perfume_recommendations(
        counts_csv="perfume_counts.csv",
        perfumes_csv="perfumes_updated.csv",
        gender=gender,
        top_k=top_k,
        counts_delimiter=",",
        perfumes_delimiter=";"
    )
    
    logger.info("Popular by count recommendations: %s", popular_perfumes)
    
    logger.info(f"Getting top rated perfumes for gender={gender}, top_k={top_k}")
     
    top_rated_perfumes = get_top_rated_recommendations(
        perfumes_csv="perfumes_updated.csv",
        gender=gender,
        top_k=top_k,
        delimiter=";",
        encoding="latin-1"
    )
     
    logger.info("Top rated recommendations: %s", top_rated_perfumes)


    content_recs_display = to_display_list(content_based_recs)
    collab_recs_display = to_display_list(collab_recs)
    popular_display = to_display_list(popular_perfumes)
    top_rated_display = to_display_list(top_rated_perfumes)

    return render_template(
        'recommendations.html',
        perfumes=user_selected,
        gender=gender,
        content_based=content_recs_display,
        collab_based=collab_recs_display,
        popular_count = popular_display,
        top_rated = top_rated_display
    )

@app.route('/recommendation-update', methods=['GET'])
def recommendation_update():
    """
    Dynamic endpoint for fetching additional recommendations based on a liked perfume.
    Expects a query parameter 'liked' (the liked perfume's name).
    Logs the request and returns exactly three new recommendations plus an alert message.
    """
    liked_perfume = request.args.get('liked')
    if not liked_perfume:
        logger.info("No liked perfume provided for update.")
        return jsonify([])

    logger.info("Fetching additional recommendations for liked perfume: %s", liked_perfume)
    new_recommendations = get_content_based_recommendations(
        selected_perfumes=[liked_perfume],
        tfidf_matrix=tfidf_matrix,
        name_to_index=name_to_index,
        index_to_name=index_to_name,
        top_k_each=200,
        final_n=3  # Return exactly 3 new recommendations.
    )
    logger.info("New recommendations for '%s': %s", liked_perfume, new_recommendations)

    new_recs_display = to_display_list(new_recommendations)
    alert_message = "3 new recommendations have been added to recommended."
    logger.info("Alert: %s", alert_message)
    # Return a JSON object with both the alert message and the new recommendations.
    return jsonify({"alert": alert_message, "recommendations": new_recs_display})

if __name__ == '__main__':
    os.makedirs('static/images', exist_ok=True)
    app.run(debug=True)
