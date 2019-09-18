#pragma once

const char vertexShader[] =
    "#version 330\n"
    "\n"
    "layout(location = 0) in vec3 position;\n"
    "layout(location = 1) in vec3 a_normal;\n"
    "uniform mat4 mvpMat;\n"
    "out vec3 shaded_colour;\n"
    "\n"
    "void main(void) {\n"
    "    gl_Position = mvpMat * vec4(position, 1.0);\n"
    "    vec3 lightpos = vec3(5, 5, 5);\n"
    "    const float ka = 0.3;\n"
    "    const float kd = 0.5;\n"
    "    const float ks = 0.2;\n"
    "    const float n = 20.0;\n"
    "    const float ax = 1.0;\n"
    "    const float dx = 1.0;\n"
    "    const float sx = 1.0;\n"
    "    const float lx = 1.0;\n"
    "    vec3 L = normalize(lightpos - position);\n"
    "    vec3 V = normalize(vec3(0.0) - position);\n"
    "    vec3 R = normalize(2 * a_normal * dot(a_normal, L) - L);\n"
    "    float i1 = ax * ka * dx;\n"
    "    float i2 = lx * kd * dx * max(0.0, dot(a_normal, L));\n"
    "    float i3 = lx * ks * sx * pow(max(0.0, dot(R, V)), n);\n"
    "    float Ix = max(0.0, min(255.0, i1 + i2 + i3));\n"
    "    shaded_colour = vec3(Ix, Ix, Ix);\n"
    "}\n";

// const char vertexShader[] =
//     "#version 330\n"
//     "\n"
//     "layout(location = 0) in vec3 position;\n"
//     "layout(location = 1) in vec3 a_normal;\n"
//     "\n"
//     "uniform mat4 mvp_matrix;\n"
//     "\n"
//     "out vec3 shaded_colour;\n"
//     "\n"
//     "void main(void) {\n"
//     "    gl_Position = vec4(position, 1.0);\n"
//     "}\n";

const char fragShader[] =
    "#version 330\n"
    "\n"
    "in vec3 shaded_colour;\n"
    "out vec4 colour_out;\n"
    "void main(void) {\n"
    "    colour_out = vec4(shaded_colour, 1);\n"
    "}\n";

// const char fragShader[] =
//     "#version 330\n"
//     "\n"
//     "void main(void) {\n"
//     "}\n";