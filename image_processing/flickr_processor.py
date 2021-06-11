import flickrapi
import ast

__api_key = u'52378bb9a95206411631fe5346f4e488'
__api_secret = u'c7a1c3c28b8d4404'

flickr = flickrapi.FlickrAPI(__api_key, __api_secret, format='json')


def search_by_tags(tags, text=None, result_type='json', result_page=1, per_page=100):
    photos_bytes = flickr.photos.search(tags=tags, text=text, page=result_page, per_page=per_page)
    photos_json = ast.literal_eval(photos_bytes.decode('utf-8'))
    if result_type == 'json':
        return photos_json['photos']['photo']
    elif result_type == 'url':
        photos_urls = []
        photos_json_arr = photos_json['photos']['photo']
        for photo_json in photos_json_arr:
            photos_urls.append(construct_url(photo_json))
        return photos_urls
    else:
        raise ValueError('Unprocessable type param value passed')


def construct_url(photo_json):
    return 'https://live.staticflickr.com/%s/%s_%s.jpg' % (photo_json['server'], photo_json['id'], photo_json['secret'])
