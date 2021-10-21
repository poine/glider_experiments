

test_doc:
	cd docs; bundle exec jekyll serve


clean:
	find . -name '*~' -exec rm {} \;
