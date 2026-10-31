-- "Handwriting cloner": click to start the dashboard and open it in the
-- browser. The app stays in the Dock while the server runs; quitting it
-- (Cmd-Q or right-click > Quit) stops the server.

property project : "/Users/blakey5aces/Handwriting Analysis"
property siteURL : "http://127.0.0.1:8765"

on serverUp()
	try
		do shell script "/usr/bin/curl -s -o /dev/null -m 2 " & siteURL & "/"
		return true
	on error
		return false
	end try
end serverUp

-- Listening, not answering: the server is single-threaded while it generates
-- words, so a busy server fails a curl but still holds its port.
on listening()
	try
		do shell script "/usr/sbin/lsof -ti tcp:8765 -sTCP:LISTEN"
		return true
	on error
		return false
	end try
end listening

on run
	if not listening() then
		-- cd first, then background only the server: backgrounding a "cd && ..."
		-- chain leaves the launching shell alive, and do shell script then waits
		-- on it forever, so the app could never answer a Quit.
		do shell script "cd " & quoted form of project & "; nohup .venv/bin/python3 dashboard/server.py > \"$HOME/Library/Logs/HandwritingCloner.log\" 2>&1 < /dev/null &"
		repeat 80 times
			delay 0.5
			if serverUp() then exit repeat
		end repeat
		if not serverUp() then
			display alert "Handwriting cloner could not start" message "The server did not come up. Details are in ~/Library/Logs/HandwritingCloner.log" as critical
			quit
			return
		end if
	end if
	open location siteURL
end run

-- clicking the Dock icon again just brings the page back
on reopen
	open location siteURL
end reopen

-- if the server dies (crash, stopped from a terminal), there is nothing left
-- for the app to represent
on idle
	if not listening() then quit
	return 5
end idle

on quit
	try
		do shell script "/bin/kill $(/usr/sbin/lsof -ti tcp:8765 -sTCP:LISTEN)"
	end try
	continue quit
end quit
