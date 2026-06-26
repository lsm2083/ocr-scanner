import java.io.FileInputStream
import java.util.Properties

plugins {
    alias(libs.plugins.android.application)
}

// Cloud Vision API key is kept out of source — paste it in local.properties as
// CLOUD_VISION_API_KEY=... (left blank => cloud OCR button shows a hint).
val cloudVisionKey: String = Properties().apply {
    val f = rootProject.file("local.properties")
    if (f.exists()) FileInputStream(f).use { load(it) }
}.getProperty("CLOUD_VISION_API_KEY", "")

android {
    namespace = "com.example.detectapp"
    compileSdk {
        version = release(37)
    }

    defaultConfig {
        applicationId = "com.example.detectapp"
        minSdk = 29
        targetSdk = 37
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        buildConfigField("String", "CLOUD_VISION_API_KEY", "\"$cloudVisionKey\"")

        ndk {
            // OpenCV maven AAR ships these ABIs. Limit to keep build/APK size sane.
            abiFilters += listOf("arm64-v8a", "armeabi-v7a", "x86_64")
        }

        externalNativeBuild {
            cmake {
                // OpenCV's prefab libs are built against the shared C++ STL.
                arguments += "-DANDROID_STL=c++_shared"
            }
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }
    buildFeatures {
        viewBinding = true
        // Exposes the OpenCV AAR's native headers/libs to our CMake build.
        prefab = true
        buildConfig = true
    }
}

dependencies {
    implementation(libs.androidx.appcompat)
    implementation(libs.androidx.constraintlayout)
    implementation(libs.androidx.core.ktx)
    implementation(libs.material)

    // OpenCV (Java API + native libs + prefab for C++/NDK linking)
    implementation(libs.opencv)

    // CameraX
    implementation(libs.androidx.camera.core)
    implementation(libs.androidx.camera.camera2)
    implementation(libs.androidx.camera.lifecycle)

    // ML Kit on-device OCR (Korean + Latin)
    implementation(libs.mlkit.text.korean)
    // ML Kit on-device translation + source-language detection
    implementation(libs.mlkit.translate)
    implementation(libs.mlkit.language.id)
    // Required so ML Kit's model downloader (MDD) can schedule its download job.
    implementation(libs.androidx.work.runtime)
    // On-device entity extraction (phone/address/date/email/url) for smart actions
    implementation(libs.mlkit.entity.extraction)

    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.espresso.core)
    androidTestImplementation(libs.androidx.junit)
}
