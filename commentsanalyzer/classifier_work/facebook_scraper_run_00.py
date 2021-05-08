from facebook_scraper import get_posts

for post in get_posts('mbctv.malawi/posts/3246777818757987', options={"comments": True}):
    print(post)
    #print(post['comments_full'][:])

