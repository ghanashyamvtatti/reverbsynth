(
	SynthDef(\help_param, { |inbufnum, outbufnum|
		var source, amp, freq, hasfreq;
		source = PlayBuf.ar(1, inbufnum, BufRateScale.kr(inbufnum));
		amp = Amplitude.kr(source);
		BufWr.kr(amp, outbufnum, Phasor.ar(0, BufRateScale.kr(outbufnum) * ControlRate.ir / SampleRate.ir, 0, BufFrames.kr(outbufnum)), 0);
		Out.ar(0, source.dup);
	}).store;
)

var inputpath = "~/Downloads/nsynth-train/bass_acoustic_examples/*.wav";
var outputpath = "~/Downloads/nsynth-train/bass_reverb_examples";
var synthdefname = \help_addalittlereverb;
var synthparams=nil;
var maxtimeperfile=0;
var outputlengthfunc=nil;
var synthdurfunc=nil;
var plot=false;
var inchannels=1;
var outchannels=1;
var outputnameadd="";
var extrainitcommands=nil;
var files, infilelength, outfilelength, sfreader, outfilepath, commands, opts, limit, check, synthdur;

// Grab a list of files matching the pattern
files = inputpath.pathMatch.collect(PathName.new(_));
if(files.size==0, {
	"Error in Score.batchNRT: No files matched input path pa```ttern.".postln;
	^nil;
});

sfreader = SoundFile.new;

outputpath = outputpath.standardizePath;

Routine({

	files.do
	{ |filepath|
		if(sfreader.openRead(filepath.fullPath).not, {
			("Error: SoundFile could not read "++filepath).postln;
		}, {
			//"DEBUG: Opened sound file".postln;
			infilelength = sfreader.numFrames / sfreader.sampleRate;

			if(sfreader.numChannels != inchannels, {
				("Warning: "++inchannels++" expected, but "++sfreader.numChannels
					++" found in file "++filepath.fileName).postln;
			});
			// Decide on the length of the output file, using the func if supplied
			outfilelength = if(outputlengthfunc.isNil, sfreader.numFrames, outputlengthfunc.value(infilelength));
			synthdur = if(synthdurfunc.isNil, infilelength, synthdurfunc.value(infilelength));
			// Decide on the output file path
			outfilepath = outputpath ++ "/" ++
			filepath.fileNameWithoutExtension ++ outputnameadd
			++ "." ++ filepath.extension;

			// Create the list of commands, which must:
			commands = extrainitcommands ++
			[
				//  - Load the input buffer
				[0.0, [\b_allocRead, 0, filepath.fullPath]],
				//  - Create the output buffer
				[0.0, [\b_alloc, 1, outfilelength, outchannels]],
				//  - Create the synth
				[0.0, [ \s_new, synthdefname, 1000, 0, 0,\inbufnum,0, \outbufnum, 1, \length, infilelength] ++ synthparams],
				// [...later...]
				//  - Write the output data to disk
				[synthdur,[\b_write, 1, outfilepath,"WAV", "float"]],
				// Kill the synth
				[synthdur, [\n_free, 1000]]
			];

			commands.postcs;

			opts = ServerOptions.new.numOutputBusChannels_(outchannels);

			// RUN THE NRT PROCESS
			("Launching NRT for file "++filepath.fileName).postln;
			Score.recordNRT(commands, nil,outfilepath, nil,16000, "WAV", "int16", opts); // synthesize

			// Now wait for it to terminate
			limit = maxtimeperfile / 0.2;
			0.2.wait;
			while({
				check="ps -xc | grep 'scsynth'".systemCmd; //256 if not running, 0 if running
				//["DEBUG: waiting for NRT", check, limit].postln;
				(check==0) and: {(limit = limit - 1) != 0} // "!=" caters for both limit and no limit
			},{
				0.2.wait;
			});
			//"DEBUG: an NRT has finished".postln;

			// Plot the output buffer if requested (would need to load from output sound file)
			if(plot, {
				var psfr; // Please ignore SC's warning about not inlining this function.
				psfr = SoundFile.new;
				psfr.openRead(outfilepath);
				{
					psfr.plot;
				}.defer;
			});
		});
	};
	// End of foreach file

	"-----------------------------------".postln;
	"Score.batchNRT process has finished".postln;
	"-----------------------------------".postln;

}).play(SystemClock);