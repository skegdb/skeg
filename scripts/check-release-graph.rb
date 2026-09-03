#!/usr/bin/env ruby
# Asserts the build/promote shape of release.yml: no job that publishes
# anything public may run before every build job has succeeded, and no
# build job may publish. actionlint checks syntax; this checks the graph.
# Run: ruby scripts/check-release-graph.rb [path/to/release.yml]
require 'yaml'

root = File.expand_path('..', __dir__)
rel = YAML.load_file(ARGV[0] || File.join(root, '.github/workflows/release.yml'))
dp  = YAML.load_file(File.join(root, '.github/workflows/docker-publish.yml'))
jobs = rel['jobs']
BUILDS = %w[guard version-gate test build-binaries docker-build].freeze
PUBLISH_MARKERS = ['action-gh-release', 'cargo publish', 'git push', 'imagetools create'].freeze

def publishes?(name, job)
  text = job.to_yaml
  return true if PUBLISH_MARKERS.any? { |m| text.include?(m) }
  job['uses'].to_s.include?('docker-publish') && job.dig('with', 'stage').to_s != 'build'
end

def ancestors(jobs, name, seen = {})
  Array(jobs[name]['needs']).each do |n|
    next if seen[n]
    seen[n] = true
    ancestors(jobs, n, seen)
  end
  seen.keys
end

fails = []
BUILDS.each { |b| fails << "missing build job #{b}" unless jobs[b] }
publishers = jobs.select { |n, j| publishes?(n, j) }.keys
fails << 'no publishing job found (markers stale?)' if publishers.empty?
publishers.each do |p|
  missing = BUILDS - ancestors(jobs, p)
  fails << "#{p} publishes but does not (transitively) need #{missing.join(', ')}" unless missing.empty?
end
(BUILDS & publishers).each { |b| fails << "build job #{b} publishes" }
# Manual runs are dry runs: every publisher must be keyed on the tag push.
# Two jobs may also run on a manual run WITH `validate_promote`, because
# there they publish only throwaway targets (a draft release, a scratch
# image tag) that `validate-cleanup` deletes; crates.io and Homebrew never.
VALIDATE_IF = "github.event_name == 'push' || inputs.validate_promote == true".freeze
VALIDATE_OK = %w[promote-release docker-promote].freeze
publishers.each do |p|
  cond = jobs[p]['if'].to_s.strip
  next if cond == "github.event_name == 'push'"
  next if VALIDATE_OK.include?(p) && cond == VALIDATE_IF
  fails << "#{p} may publish on a manual run (if: #{cond.inspect})"
end
fails << 'validate-cleanup must exist and need both promote jobs' unless jobs['validate-cleanup'] && (%w[promote-release docker-promote] - Array(jobs['validate-cleanup']['needs'])).empty?
fails << 'docker-promote must pass a scratch tag on manual runs' unless jobs['docker-promote'].dig('with', 'scratch_tag').to_s.include?('validate-')
fails << 'build-binaries is not a matrix build' unless jobs['build-binaries'].dig('strategy', 'matrix')

# docker-publish.yml: `stage: build` must not tag, `stage: promote` must not rebuild.
fails << "docker-publish build job does not skip on stage=promote" unless dp.dig('jobs', 'build', 'if').to_s.include?("inputs.stage != 'promote'")
fails << "docker-publish merge job does not skip on stage=build" unless dp.dig('jobs', 'merge', 'if').to_s.include?("inputs.stage != 'build'")
fails << "docker-publish still fires on v* tags by itself" if Array(dp.dig(true, 'push', 'tags') || dp.dig('on', 'push', 'tags')).any?

if fails.empty?
  puts "ok: publishers #{publishers.join(', ')} all wait for #{BUILDS.join(', ')}"
else
  fails.each { |f| warn "FAIL: #{f}" }
  exit 1
end
