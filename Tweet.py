class Tweet:
    id = 0
    posted_by = ''
    talk_about = ''
    text = ''
    description = ''
    time_zone = ''
    user_id = ''
    coordinates = ''
    tweets_per_user = ''
    created_at = ''
    screen_name = ''

    def __init__(self, tweet_text=None):
        if tweet_text is not None:
            segments = str(tweet_text).split('\t')

            self.id = segments[0]
            self.posted_by = segments[1]
            self.talk_about = segments[2]
            self.text = segments[3]
            self.description = segments[4]
            self.time_zone = segments[5]
            self.user_id = segments[6]
            self.coordinates = segments[7]
            self.tweets_per_user = segments[8]
            self.created_at = segments[9]
            self.screen_name = segments[11]