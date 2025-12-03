#include "GPUConvertService.h"

GLMesh GPUConvertService::uploadMesh(const Mesh& m) {
	GLMesh gm;
	if (m.vertices.empty() || m.indices.empty()) return gm;

	gm.indexCount = (GLsizei)m.indices.size();

	// Compute AABB
	glm::vec3 minB(
		std::numeric_limits<float>::max(),
		std::numeric_limits<float>::max(),
		std::numeric_limits<float>::max());
	glm::vec3 maxB(
		std::numeric_limits<float>::lowest(),
		std::numeric_limits<float>::lowest(),
		std::numeric_limits<float>::lowest());

	size_t vertexCount = m.vertices.size() / 3;
	for (size_t i = 0; i < vertexCount; ++i) {
		glm::vec3 v(
			m.vertices[i * 3 + 0],
			m.vertices[i * 3 + 1],
			m.vertices[i * 3 + 2]);
		minB = glm::min(minB, v);
		maxB = glm::max(maxB, v);
	}
	gm.aabbMin = minB;
	gm.aabbMax = maxB;

	glGenVertexArrays(1, &gm.vao);
	glGenBuffers(1, &gm.vbo);
	glGenBuffers(1, &gm.ebo);

	glBindVertexArray(gm.vao);

	glBindBuffer(GL_ARRAY_BUFFER, gm.vbo);
	glBufferData(GL_ARRAY_BUFFER, m.vertices.size() * sizeof(float),
		m.vertices.data(), GL_STATIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gm.ebo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, m.indices.size() * sizeof(unsigned int),
		m.indices.data(), GL_STATIC_DRAW);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);

	glBindVertexArray(0);
	return gm;
}