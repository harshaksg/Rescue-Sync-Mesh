package com.example.rescuesync.ai

import android.content.Context
import org.tensorflow.lite.task.audio.classifier.AudioClassifier
import java.util.Timer
import java.util.TimerTask

class EmergencyDetector(private val context: Context, private val listener: DetectionListener) {

    // 1. Define the interface to talk to the rest of the app
    interface DetectionListener {
        fun onEmergencyDetected(label: String, probability: Float)
    }

    private var classifier: AudioClassifier? = null
    private var timer: Timer? = null

    init {
        // 2. Load the model from the assets folder
        try {
            classifier = AudioClassifier.createFromFileAndOptions(
                context,
                "yamnet.tflite",
                AudioClassifier.AudioClassifierOptions.builder().build()
            )
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    fun startListening() {
        val audioRecord = classifier?.createAudioRecord()
        audioRecord?.startRecording()

        timer = Timer()
        timer?.scheduleAtFixedRate(object : TimerTask() {
            override fun run() {
                // 3. Capture audio and run inference
                val tensorAudio = classifier?.createInputTensorAudio()
                tensorAudio?.load(audioRecord)

                val results = classifier?.classify(tensorAudio)

                // 4. Filter for Distress Sounds (Screaming = Index 11, Whistle = Index 396)
                results?.get(0)?.categories?.forEach { category ->
                    if ((category.label == "Screaming" || category.label == "Whistle")
                        && category.score > 0.5f) {
                        listener.onEmergencyDetected(category.label, category.score)
                    }
                }
            }
        }, 0, 1000) // Runs every 1 second
    }

    fun stopListening() {
        timer?.cancel()
        classifier?.close()
    }
}