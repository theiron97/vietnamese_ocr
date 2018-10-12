#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf8')
from keras import applications



if __name__ == "__main__":
    base_model = applications.VGG16(weights='imagenet', include_top=False)
    